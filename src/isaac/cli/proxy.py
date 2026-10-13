import typer

app = typer.Typer()


@app.command(name="placeholder")
def placeholder():
    print("This is the proxy sub-app. Logic migrated from monolith.")
