"""Allow ``python -m vector_cli`` as an alias for the ``vectorise`` command."""

from vector_cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
