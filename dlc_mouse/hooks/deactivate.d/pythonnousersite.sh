# Restore previous value (or unset if it was not set)
if [ "${_OLD_PYTHONNOUSERSITE-__UNSET__}" = "__UNSET__" ]; then
  unset PYTHONNOUSERSITE
else
  export PYTHONNOUSERSITE="${_OLD_PYTHONNOUSERSITE}"
fi
unset _OLD_PYTHONNOUSERSITE
