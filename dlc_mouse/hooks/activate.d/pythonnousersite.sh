# Save previous value (or sentinel if unset), then force env isolation
export _OLD_PYTHONNOUSERSITE="${PYTHONNOUSERSITE-__UNSET__}"
export PYTHONNOUSERSITE=1
