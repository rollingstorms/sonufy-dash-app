import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

from config import config



########################## Setup ##########################

# App Instance
app = dash.Dash(name=config.name, assets_folder="static", external_stylesheets=[dbc.themes.LUX, config.font_awesome])
app.title = config.name

########################## Header ##########################



########################## Body ##########################



########################## Layout ##########################


app.layout = dbc.Container(fluid=True, children=[
	html.H1(config.name, id="nav-pills")
    
])


########################## Run ##########################
if __name__ == "__main__":
    debug = config.debug
    app.run_server(debug=debug, host=config.host, port=config.port)