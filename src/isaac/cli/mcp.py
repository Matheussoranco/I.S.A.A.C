import typer

app = typer.Typer()


@app.command(name="placeholder")
def placeholder():
    print("This is the mcp sub-app. Logic migrated from monolith.")
