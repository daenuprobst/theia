from theia.web import create_app


def local():
    app = create_app()
    app.run()
