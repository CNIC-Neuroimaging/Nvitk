"""Minimal JPEG 2000 packet-header walker: enough to locate packet boundaries.

Analysis tool, not part of the conversion path. It exists for the Zeiss OCT series whose
private payload is stored obfuscated (LuraWave-encoded, every seventh byte XORed with 0x5A,
and the JP2 file's parts reordered), where locating real packet boundaries is the only way
to test a candidate reconstruction without a decoder.

Covers the profile those streams use: one tile, one tile-part, one layer, no precincts, no
SOP/EPH markers and a single codeword segment per code-block. Validated against a codestream
encoded with identical parameters, where it walks all six packets and lands exactly on the
end of the tile-part.

--------------------------------------------------------------------------------------------
Do not use in practice! The output volumes will produce visually plausible images, but not real!
--------------------------------------------------------------------------------------------
"""
import struct

def cdiv(a, b): return -((-a) // b)

class BitReader:
    """Packet-header bit reader with JPEG 2000 bit-unstuffing (a byte after 0xFF gives 7 bits)."""
    def __init__(self, buf, pos=0):
        self.buf, self.pos, self.ctx, self.n, self.prev_ff = buf, pos, 0, 0, False
    def bit(self):
        if self.n == 0:
            if self.pos >= len(self.buf): raise EOFError
            b = self.buf[self.pos]; self.pos += 1
            if self.prev_ff:
                if b > 0x7F: raise ValueError("stuffing violado")
                self.ctx, self.n = b, 7
            else:
                self.ctx, self.n = b, 8
            self.prev_ff = (b == 0xFF)
        self.n -= 1
        return (self.ctx >> self.n) & 1
    def bits(self, k):
        v = 0
        for _ in range(k): v = (v << 1) | self.bit()
        return v
    def align(self):
        """Finish the header: drop remaining bits, and consume the stuffed byte after a final 0xFF."""
        self.n = 0
        if self.prev_ff:
            if self.pos >= len(self.buf): raise EOFError
            if self.buf[self.pos] > 0x7F: raise ValueError("stuffing final violado")
            self.pos += 1
            self.prev_ff = False

class TagTree:
    def __init__(self, w, h):
        self.w, self.h, self.levels = w, h, []
        while True:
            self.levels.append({"w": w, "h": h, "val": [0]*(w*h), "known": [False]*(w*h)})
            if w == 1 and h == 1: break
            w, h = cdiv(w, 2), cdiv(h, 2)
    def decode(self, br, x, y, threshold):
        """Standard tag-tree decode; returns the node value (capped at *threshold* when not resolved)."""
        lo = 0
        for lv in range(len(self.levels)-1, -1, -1):
            L = self.levels[lv]
            xx, yy = x >> lv, y >> lv
            i = yy*L["w"] + xx
            if L["val"][i] < lo: L["val"][i] = lo
            while not L["known"][i] and L["val"][i] < threshold:
                if br.bit(): L["known"][i] = True
                else: L["val"][i] += 1
            lo = L["val"][i]
            if not L["known"][i]: return lo, False
        return lo, True

def bands_for(res, NL, tcx0, tcy0, tcx1, tcy1):
    """Sub-band rectangles contributing to one resolution level."""
    if res == 0:
        nb = NL
        return [(cdiv(tcx0, 1 << nb), cdiv(tcy0, 1 << nb), cdiv(tcx1, 1 << nb), cdiv(tcy1, 1 << nb))]
    nb = NL - res + 1
    out = []
    for xob, yob in ((1, 0), (0, 1), (1, 1)):
        o = 1 << (nb - 1)
        out.append((cdiv(tcx0 - o*xob, 1 << nb), cdiv(tcy0 - o*yob, 1 << nb),
                    cdiv(tcx1 - o*xob, 1 << nb), cdiv(tcy1 - o*yob, 1 << nb)))
    return out

def npasses(br):
    if br.bit() == 0: return 1
    if br.bit() == 0: return 2
    v = br.bits(2)
    if v < 3: return 3 + v
    v = br.bits(5)
    if v < 31: return 6 + v
    return 37 + br.bits(7)

class PacketWalker:
    """Walks packets in LRCP order, returning each packet's byte extent."""
    def __init__(self, W, H, NL, layers, xcb, ycb, ppx=15, ppy=15):
        self.W, self.H, self.NL, self.layers = W, H, NL, layers
        self.state = {}
        for r in range(NL+1):
            for bi, (bx0, by0, bx1, by1) in enumerate(bands_for(r, NL, 0, 0, W, H)):
                # xcb' is capped by the precinct: PPx for r=0, PPx-1 above it.
                cbw = min(xcb, ppx if r == 0 else ppx-1)
                cbh = min(ycb, ppy if r == 0 else ppy-1)
                nx = 0 if bx1 <= bx0 else (((bx1-1) >> cbw) - (bx0 >> cbw) + 1)
                ny = 0 if by1 <= by0 else (((by1-1) >> cbh) - (by0 >> cbh) + 1)
                self.state[(r, bi)] = {
                    "nx": nx, "ny": ny, "n": nx*ny,
                    "incl": TagTree(max(nx,1), max(ny,1)), "imsb": TagTree(max(nx,1), max(ny,1)),
                    "included": [False]*(nx*ny), "lblock": [3]*(nx*ny),
                }
    def packet(self, buf, pos, res, layer):
        """Parse one packet header at *pos*; returns (header_end, body_len)."""
        br = BitReader(buf, pos)
        if br.bit() == 0:
            br.align()
            return br.pos, 0
        body = 0
        for bi, _ in enumerate(bands_for(res, self.NL, 0, 0, self.W, self.H)):
            st = self.state[(res, bi)]
            for k in range(st["n"]):
                x, y = k % st["nx"], k // st["nx"]
                if st["included"][k]:
                    inc = br.bit() == 1
                else:
                    v, done = st["incl"].decode(br, x, y, layer+1)
                    inc = done and v <= layer
                if not inc: continue
                if not st["included"][k]:
                    st["included"][k] = True
                    t = 1
                    while True:
                        v, done = st["imsb"].decode(br, x, y, t)
                        if done: break
                        t += 1
                np_ = npasses(br)
                while br.bit() == 1: st["lblock"][k] += 1
                nbits = st["lblock"][k] + np_.bit_length() - 1
                body += br.bits(nbits)
        br.align()
        return br.pos, body
    def walk(self, buf, pos=0, limit=None):
        """Yield (layer, res, header_start, header_end, body_len) in LRCP order."""
        out = []
        for layer in range(self.layers):
            for res in range(self.NL+1):
                h0 = pos
                h1, body = self.packet(buf, pos, res, layer)
                pos = h1 + body
                out.append((layer, res, h0, h1, body))
                if limit is not None and pos > limit: return out
        return out
