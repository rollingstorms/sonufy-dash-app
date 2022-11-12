from dash_auth_external import DashAuthExternal
from dash import Dash, Input, Output, html, dcc
import dash

import flask

# using spotify as an example
AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
CLIENT_ID = "6ecc85041d924931a24d8e59e50e03f9"

# creating the instance of our auth class
auth = DashAuthExternal(AUTH_URL, TOKEN_URL, CLIENT_ID)
server = (
    auth.server
)  # retrieving the flask server which has our redirect rules assigned


app = Dash(__name__, server=server, use_pages=True)  # instantiating our app using this server


##Below we can define our dash app like normal
app.layout = html.Div([html.Div(id="example-output"), dash.page_container, dcc.Input(id="example-input")])


@app.callback(Output("example-output", "children"), Input("example-input", "value"))
def example_callback(value):
    print(auth.get_headers())
    print(app.server)
    token = (
        auth.get_token()
    )  ##The token can only be retrieved in the context of a dash callback
    return token


if __name__ == "__main__":
    app.run_server(debug=True, host='127.0.0.1', port='8050')