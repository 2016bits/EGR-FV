#!/bin/sh

resolve_repo_root() {
  script_dir=${SCRIPT_DIR:-$(CDPATH= cd "$(dirname "$0")" && pwd)}
  cd "$script_dir/.." && pwd
}

python_has_required_modules() {
  python_bin=$1
  "$python_bin" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

required = ["torch", "transformers", "yaml", "sklearn", "tqdm"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
sys.exit(0 if not missing else 1)
PY
}

resolve_python_bin() {
  requested_python=${PYTHON_BIN:-python}
  current_user=${USER:-}
  home_dir=${HOME:-}
  seen=:

  for preferred_env in \
    "$home_dir/.conda/envs/tor230/bin/python" \
    "/data/$current_user/.conda/envs/tor230/bin/python" \
    "/data/yangjun/.conda/envs/tor230/bin/python" \
    "$home_dir/.conda/envs/egrfv/bin/python" \
    "/data/$current_user/.conda/envs/egrfv/bin/python" \
    "/data/yangjun/.conda/envs/egrfv/bin/python"; do
    for candidate in "$requested_python" "$preferred_env"; do
      case "$candidate" in
        */*)
          [ -x "$candidate" ] || continue
          resolved=$candidate
          ;;
        *)
          resolved=$(command -v "$candidate" 2>/dev/null) || continue
          ;;
      esac

      case "$seen" in
        *:"$resolved":*) continue ;;
      esac
      seen=$seen$resolved:

      if python_has_required_modules "$resolved"; then
        printf '%s\n' "$resolved"
        return 0
      fi
    done
  done

  for root in "$home_dir/.conda/envs" "/data/$current_user/.conda/envs" "/data/yangjun/.conda/envs"; do
    [ -d "$root" ] || continue
    for candidate in "$root"/*/bin/python; do
      [ -e "$candidate" ] || continue
      [ -x "$candidate" ] || continue
      resolved="$candidate"

      case "$seen" in
        *:"$resolved":*) continue ;;
      esac
      seen=$seen$resolved:

      if python_has_required_modules "$resolved"; then
        printf '%s\n' "$resolved"
        return 0
      fi
    done
  done

  echo "Unable to locate a Python interpreter with required packages: torch, transformers, PyYAML, scikit-learn, tqdm." >&2
  echo "Set PYTHON_BIN=/path/to/python or install dependencies with: pip install -r requirements.txt" >&2
  return 1
}
