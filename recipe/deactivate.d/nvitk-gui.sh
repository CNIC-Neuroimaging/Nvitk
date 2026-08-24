#!/bin/bash
if [ -n "${_NVITK_OLD_LD_LIBRARY_PATH+set}" ]; then
  export LD_LIBRARY_PATH="${_NVITK_OLD_LD_LIBRARY_PATH}"
  unset _NVITK_OLD_LD_LIBRARY_PATH
fi
