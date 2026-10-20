#!/bin/bash
if [ -n "${_NVITK_OLD_LD_LIBRARY_PATH+set}" ]; then
  export LD_LIBRARY_PATH="${_NVITK_OLD_LD_LIBRARY_PATH}"
  unset _NVITK_OLD_LD_LIBRARY_PATH
fi
if [ -n "${_NVITK_OLD_QT_PLUGIN_PATH+set}" ]; then
  if [ -n "${_NVITK_OLD_QT_PLUGIN_PATH}" ]; then
    export QT_PLUGIN_PATH="${_NVITK_OLD_QT_PLUGIN_PATH}"
  else
    unset QT_PLUGIN_PATH
  fi
  unset _NVITK_OLD_QT_PLUGIN_PATH
fi
if [ -n "${_NVITK_OLD_FASTSURFER_HOME+set}" ]; then
  if [ -n "${_NVITK_OLD_FASTSURFER_HOME}" ]; then
    export FASTSURFER_HOME="${_NVITK_OLD_FASTSURFER_HOME}"
  else
    unset FASTSURFER_HOME
  fi
  unset _NVITK_OLD_FASTSURFER_HOME
fi
