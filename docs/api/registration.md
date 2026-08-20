# Registration

`nvitk.registration` wraps three registration engines behind a consistent `Image`-in,
`Image`-out interface, each also exposed as a CLI entry point.

| Module / command | Engine | Purpose |
|---|---|---|
| `nvitk.registration.fsl.flirt` / `nvitk-flirt` | FSL FLIRT | Rigid/affine registration — also the engine behind the {doc}`QVTPy pipeline's <../pipelines/qvtpy>` stage-2 eICAB-to-4D-flow alignment. |
| `nvitk.registration.ants` / `nvitk-ants` | ANTsPy | Deformable/SyN registration. |
| `nvitk.registration.fireants` / `nvitk-fireants` | FireANTs | GPU-accelerated diffeomorphic registration. |

```{note}
FLIRT requires a working FSL installation on `PATH` (see {doc}`../installation`) — nvitk
wraps it via `nipype`'s FSL interface rather than reimplementing it.
```

```{seealso}
Full generated reference: [`nvitk.registration`](../autoapi/nvitk/registration/index).
```
