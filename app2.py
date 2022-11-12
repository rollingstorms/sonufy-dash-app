import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash_daq as daq

from config import config
import pandas as pd

from src.Sonufy import *

import visdcc

import plotly_express as px
import plotly.graph_objs as go

from pyarrow import feather

import requests

import spotipy

from time import sleep

BASE_URL = 'https://api.spotify.com/v1/'
AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
CLIENT_ID = '6ecc85041d924931a24d8e59e50e03f9'

SCOPE = "user-read-private user-read-playback-state user-modify-playback-state streaming"


from dash_auth_external import DashAuthExternal

auth = DashAuthExternal(AUTH_URL, TOKEN_URL, CLIENT_ID, scope=SCOPE)
server = (
    auth.server
)

def get_authorization_header(access_token):
    return {"Authorization": "Bearer {}".format(access_token)}

########################## Setup ##########################

# App Instance
app = dash.Dash(name=config.name, use_pages=True, assets_folder="static", external_scripts=['https://code.jquery.com/jquery-3.6.1.slim.min.js'], external_stylesheets=[dbc.themes.YETI], server= server, suppress_callback_exceptions=True)
app.title = config.name


def load_sonufy():
	model_path = 'model'
	sonufy = Sonufy(latent_dims=64, output_size=(64,64))
	sonufy.load_encoder(model_path)
	sonufy.load_db(model_path)

	return sonufy

sonufy = load_sonufy()

#### UMAP SETUP #####

base_genres_names = feather.read_feather('data/base_genres_names.feather')

base_genres = sonufy.genres[sonufy.genres.genre.isin(base_genres_names.name)].reset_index(drop=True)

base_genres['name'] = base_genres['genre']
base_genres['label'] = 0

genre_map = load('model/genre_map.bin')

features_to_use = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

uri = '2Amr61JmOUVaunLKPSe39i'


########################## Layout ##########################



app.layout = html.Div(id='main', children=[
	dcc.Store(id='memo'),
	html.Div(id='body', className='body', children=[

#### SEARCH #####

	html.Div(id='sonufy_nav', className='sixteen open', children=[

		html.Div(id='nav_hover', className='hover'),

		html.Div(id='nav_header', children=[
			html.Div(id='basic_nav', children=[
			# logo
			html.Div(id='logo_header', children=[
				html.Img(id='logo_img', src='logo.svg'),
				html.Img(id='title_img', src='sonufy.svg', alt='Sonufy')
			]),
		
			# search bar
			dbc.Input(id="search_input", placeholder="Search for a track on Spotify", type="text", debounce=True),
			html.Button(children='Play', id='play_button', className="togglePlay", value=uri),
			]),
		

			html.Div(id='advanced_nav', children=[
				html.Div(id='umap_nav', children=[
					dcc.Graph(id='advanced_umap', responsive=True, config={'autosizable':True, 'frameMargins':0, 'displayModeBar':'hover'})
						]),

				html.Div(id='advanced_knobs', children=[
					html.Div(className='knobs_section', children=[
						dcc.Input(id=f'number_{j*len(sonufy.latent_cols)//4 + i}', className='advanced_num_input', step=.1, type='range', inputMode='numeric', min=-1, max=1, value=0, debounce=True) for i in range(len(sonufy.latent_cols)//4)
						]) for j in range(4)
					
					])
				])
				]),

			dcc.Tabs(id='nav_tabs', value='basic', children=[
		        dcc.Tab(label='Basic', value='basic'),
		        dcc.Tab(label='Advanced', value='advanced'),
		    ]),

		html.Div(id='close_nav', className='close no_display')
		]),

	#### QUERY SONG #####

	html.Div(id='this_song', className='sixteen', children=[
		html.Div(id='this_hover', className='hover'),
		html.Div(id='this_song_div', className='track_div', children=[
		html.Div(id='this_header_div', className='this_header_div', children=[
				
			# number
			# html.H1('this song', className='artist_info'),

			# html.Img(className='cover_art', src=track['album']['images'][0]['url']),

			# title and artist
			html.P(className='artist_info', children=[
				# html.Span(track['name'], className='track_name'), 
				' by ',
				# html.Span(track['artists'][0]['name'], className='artist')
				]),
			]),

			html.Div(id='this_full', className='this_full', children=[
				html.Div(id='this_player', className='this_player', children=[
					# play button
					# like button
					]),
				
				html.Div(id='this_info', className='this_info', children=[
					# vector similarity
					html.Div(id='this_vector_compare', className='this_vector_compare', children=[

						]),
					# similarity indexes
					html.Div(id='this_similarity', className='this_similarity', children=[

						]),

					# audio feature comparison
					html.Div(id='this_feature_compare', className='this_feature_compare', children=[

						])

					])
				])
		]),
		html.Div(id='close_this', className='close')
	]),

	#### RECOMMENDATION TEMPLATE #####

	*[html.Div(id=f'rec_{i}', className='sixteen recs', children=[
		html.Div(id=f'rec_{i}_hover', className='hover'),
		html.Div(id=f'rec_div_{i}', className='track_div', children=[

			
			html.Div(id=f'rec_header_div_{i}', className='rec_header_div', children=[
			# 	# number
				html.H1(i),

				# html.Img(className='cover_art', src=latents.loc[i-1, 'album_art_url']),

			# 	# title and artist
				# html.P(className='artist_info', children=[html.Span(latents.loc[i-1 ,'track_name']), ' by ',html.Span(latents.loc[i-1 ,'artist_name'], className='artist')]),
				]),

			html.Div(id=f'rec_full_{i}', className='rec_full', children=[
				html.Div(id=f'rec_player_{i}', className='rec_player', children=[
					# html.Button(children='Play', id=f'play_button_rec_{i}', className="togglePlay", value=latents.loc[i-1, 'track_id']),

					]),
				
				html.Div(id=f'rec_info_{i}', className='rec_info', children=[
					# vector similarity
					html.Div(id=f'rec_vector_compare_{i}', className='rec_vector_compare', children=[
						html.Div(className='rec_vector vector', children=[
							# html.Div(str(round(col, 2)), className='vector_cell', style={'background':f'rgba({100*col},0,{-100*col},1)'}) for col in latents.loc[i-1, sonufy.latent_cols]
							]),

						html.Div(className='query_vector vector', children=[
							# html.Div(str(round(col, 2)), className='vector_cell', style={'background':f'rgba({100*col},0,{-100*col},1)'}) for col in list(*this_track.values)
							])
							
						]),
					# similarity indexes
					html.Div(id=f'rec_similarity_{i}', className='rec_similarity', children=[
						html.Div(className='total_similarity similarity_index', children=[
							# 'similarity: ' + str(round(latents.loc[i-1, 'similarity'], 2))
							]),
						html.Div(className='time_similarity similarity_index', children=[
							# 'time similarity: ' + str(round(latents.loc[i-1, 'time_similarity'],2))
							]),
						html.Div(className='freq_similarity similarity_index', children=[
							# 'frequency similarity: ' + str(round(latents.loc[i-1, 'frequency_similarity'],2))
							])
						]),

					# audio feature comparison
					html.Div(id=f'rec_feature_compare_{i}', className='rec_feature_compare', children=[
							html.Div(className=f'feature_compare feature_compare_title', children=[
								html.Div(className='query_feature query_feature_title', children=[
									'Query'
									]),

								html.Div(className='diff_feature diff_feature_title', children=[
									'Feature Difference'
									]),

								html.Div(className='rec_feature rec_feature_title', children=[
									'Recommendation'
									]),
								]),

							*[html.Div(className=f'feature_compare {feature}_compare', children=[
								html.Div(className='query_feature', children=[
									# audio_features[0][feature]
									]),

								html.Div(className='diff_feature', children=[
									# html.Div(feature, className='diff_title'),
									# html.Div(round(abs(audio_features[0][feature] - audio_features[i][feature]),2), className='diff_quant')
									]),

								html.Div(className='rec_feature', children=[
									# audio_features[i][feature]
									]),
								]) for feature in features_to_use]
						])

					])

				])
		]),
		html.Div(id=f'close_rec_{i}', className='close')
		]) for i in range(1,11)],

	#### UMAP PLOT #####

	html.Div(id='umap_plot', className='umap_plot', children=[
		html.H1('Umap Plot')
		])
	]),
	dash.page_container
])

########################## CALLBACKS ##########################


@app.callback(
	Output('advanced_umap', 'figure'),
	*[Input(f'number_{i}', 'value') for i in range(len(sonufy.latent_cols))]
)
def render_advanced_umap(*numbers):
	this_track_df = pd.DataFrame([numbers], columns=sonufy.latent_cols)
	this_track_df['name'] = 'input'
	this_track_df['label'] = 2

	genres_and_tracks = pd.concat([base_genres, this_track_df]).reset_index(drop=True)

	genre_map_trans = genre_map.transform(genres_and_tracks[sonufy.latent_cols])

	genre_map_df = pd.DataFrame(genre_map_trans, columns=['x','y'])
	genre_map_df = pd.concat([genres_and_tracks[['name','label']], genre_map_df], axis=1)
	genre_map_df.label = genre_map_df.label.map({0:'genre', 1:'similar song', 2:'this song'})
	genre_map_df['annotation'] = genre_map_df.apply(lambda x: x['name'] if x['label'] == 'genre' else '', axis=1)

	fig = px.scatter(genre_map_df, x='x', y='y', color='label', hover_name='name', size=[.5]*len(genre_map_df), width=800, height=600, text='annotation')
	

	fig.update_layout(
	    margin = dict(l=0, r=0, t=0, b=0),
	    legend=dict(
		    yanchor="top",
		    y=0.99,
		    xanchor="left",
		    x=0.01
		),
		xaxis = dict(visible=False),
		yaxis = dict(visible=False)
	)

	return fig

@app.callback(
	Output('basic_nav', 'className'),
	Output('advanced_nav', 'className'),
	Input('nav_tabs', 'value'),
)
def render_content(tab):
	if tab == 'basic':
		return 'show', 'noshow'
	if tab == 'advanced':
		return 'noshow', 'show'

#player example code

@app.callback(
	Output('play_button', 'children'),
	Input('play_button', 'n_clicks'),
	Input('play_button', 'value'))
def test_spotify(n_clicks, uri):
	if n_clicks != None:
		token = (auth.get_token())

		spotify = spotipy.Spotify(auth=token)
		devices = spotify.devices()

		device_id = devices['devices'][0]['id']

		play_uri = [f'spotify:track:{uri}']

		current_track = spotify.current_user_playing_track()


		if current_track == None:
			spotify.start_playback(device_id=device_id, uris=play_uri)
			print('started from scratch, now playing')
			return 'Pause'
		else:
			playing = current_track['is_playing']
			current_uri = current_track['item']['id']

			if playing and current_uri == uri:
				spotify.pause_playback()
				print('paused')
				return 'Play'
			elif current_uri != uri:
				spotify.start_playback(uris=play_uri)
				print('played from start')
				return 'Pause'
			else:
				spotify.start_playback()
				print('resumed play')
				return 'Pause'

	else:
		return 'Play'


#search

@app.callback(
	Output('sonufy_nav', 'className'),
	Output('close_nav', 'className'),
	Output('this_song_div', 'children'),
	*[Output(f'rec_div_{i}', 'children') for i in range(1,11)],
	*[Output(f'play_button_rec_{i}', 'value') for i in range(1,11)],
	Output('umap_plot', 'children'),
	*[Output(f'number_{i}', 'value') for i in range(len(sonufy.latent_cols))],
	Input(component_id='search_input', component_property='value')
	)
def search(search_input):

	if search_input is None or '':
		raise PreventUpdate

	# try:
	track, latents, this_track, audio_features = sonufy.search_for_recommendations(search_input, get_time_and_freq=True)
	
	features_to_use = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
	

	this_song_layout = 

	recs_layout = []

	for i in range(1,11):

		this_rec_layout = 
		recs_layout.append(this_rec_layout)

		@app.callback(
			Output(f'play_button_rec_{i}', 'children'),
			Input(f'play_button_rec_{i}', 'n_clicks'),
			Input(f'play_button_rec_{i}', 'value'))
		def test_spotify(n_clicks, uri):
			if uri == None:
				PreventUpdate()

			if n_clicks != None:
				token = (auth.get_token())

				spotify = spotipy.Spotify(auth=token)
				devices = spotify.devices()

				device_id = devices['devices'][0]['id']

				play_uri = [f'spotify:track:{uri}']

				current_track = spotify.current_user_playing_track()


				if current_track == None:
					spotify.start_playback(device_id=device_id, uris=play_uri)
					print('started from scratch, now playing')
					return 'Pause'
				else:
					playing = current_track['is_playing']
					current_uri = current_track['item']['id']

					if playing and current_uri == uri:
						spotify.pause_playback()
						print('paused')
						return 'Play'
					elif current_uri != uri:
						spotify.start_playback(uris=play_uri)
						print('played from start')
						return 'Pause'
					else:
						spotify.start_playback()
						print('resumed play')
						return 'Pause'

			else:
				return 'Play'

	######## UMAP ###########


	# see top for umap setup

	this_track_df = pd.DataFrame(this_track, columns=sonufy.latent_cols)
	this_track_df['name'] = track['name'] + ' - ' + track['artists'][0]['name']
	this_track_df['label'] = 2

	latents['name'] = latents['track_name'] + ' - ' + latents['artist_name']
	latents['label'] = 1
	latents_new = latents[['name'] + sonufy.latent_cols + ['label']]

	genres_and_tracks = pd.concat([base_genres, latents_new, this_track_df]).reset_index(drop=True)

	genre_map_trans = genre_map.transform(genres_and_tracks[sonufy.latent_cols])

	genre_map_df = pd.DataFrame(genre_map_trans, columns=['x','y'])
	genre_map_df = pd.concat([genres_and_tracks[['name','label']], genre_map_df], axis=1)
	genre_map_df.label = genre_map_df.label.map({0:'genre', 1:'similar song', 2:'this song'})
	genre_map_df['annotation'] = genre_map_df.apply(lambda x: x['name'] if x['label'] == 'genre' else '', axis=1)

	fig = px.scatter(genre_map_df, x='x', y='y', color='label', hover_name='name', size=[.5]*len(genre_map_df), width=800, height=600, text='annotation')
	

	fig.update_layout(
	    margin = dict(l=0, r=0, t=0, b=0),
	    legend=dict(
		    yanchor="top",
		    y=0.99,
		    xanchor="left",
		    x=0.01
		),
		xaxis = dict(visible=False),
		yaxis = dict(visible=False)
	)

	umap_layout = [
		dcc.Graph(figure=fig, responsive=True, config={'autosizable':True, 'frameMargins':0, 'displayModeBar':'hover'})
	]
	
	print(len(['sixteen', 'close', this_song_layout, *recs_layout, *list(latents.loc[:, 'track_id']), umap_layout, *this_track.values[0]]))

	return 'sixteen', 'close', this_song_layout, *recs_layout, *list(latents.loc[:, 'track_id']), umap_layout, *this_track.values[0]


	# except:
	# 	print('error')
	# 	pass




# 	token_header = get_authorization_header(token)

# 	top_url = BASE_URL + 'me/top/tracks'

# 	r = requests.get(top_url, headers=token_header)

# 	return [str(r.json()['items'][i]['name']) for i in range(len(r.json()['items']))]


########################## Run ##########################
if __name__ == "__main__":
    debug = config.debug
    app.run_server(debug=debug, host=config.host, port=config.port)