from pathlib import Path
import webbrowser


def main() -> None:
    page = Path(__file__).resolve().parent / "visualizations" / "visualization.html"
    if not page.exists():
        raise FileNotFoundError(f"Visualization page not found: {page}")
    opened = webbrowser.open(page.as_uri(), new=2)
    print(f"Визуализация: {page}")
    if not opened:
        print("Браузер не открылся автоматически — откройте этот файл вручную.")


if __name__ == "__main__":
    main()
