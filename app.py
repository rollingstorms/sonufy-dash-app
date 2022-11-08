import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from config import config
import pandas as pd

from src.Sonufy import *

import visdcc

########################## Setup ##########################

# App Instance
app = dash.Dash(name=config.name, assets_folder="static", external_scripts=['https://code.jquery.com/jquery-3.6.1.slim.min.js'], external_stylesheets=[dbc.themes.YETI])
app.title = config.name

def load_sonufy():
	model_path = 'model'
	sonufy = Sonufy(latent_dims=64, output_size=(64,64))
	sonufy.load_encoder(model_path)
	sonufy.load_db(model_path)

	return sonufy

sonufy = load_sonufy()

########################## Sonufy Nav Components ##########################

# start with 'open' className for css to display full screen

sonufy_block = []

# logo

block_header = html.Div(id='nav_header', children=[])
logo_header = html.Div(id='logo_header', children=[])

logo = html.Img(id='logo_img', src='static/logo.svg')
logo_header.children.append(logo)

# title

title = html.Img(id='title_img', src='static/sonufy.svg', alt='Sonufy')
logo_header.children.append(title)
block_header.children.append(logo_header)

# search bar

search_bar = dbc.Input(id="search_input", placeholder="Search for a track on Spotify", type="text", debounce=True)
block_header.children.append(search_bar)

	

# search button

# search_button = html.Button('Search', id='search_submit', n_clicks=0)
# block_header.append(search_button)

sonufy_block.append(block_header)

block_full = html.Div(id='nav_full', children=[])
sonufy_block.append(block_full)

# tabs?

	# advanced

		# manual search

			# umap plot

			# n dials

	# about

		# markdown of explanation?

########################## This Song Components ##########################

# This song

# album cover

# play button

# like button

# title

# artist

# year - predicted year with linear regression model on embeddings?

# genre - predict with log reg?

# how about all spotify metrics? triggered by search


########################## Recommendation Components ##########################

# x close or back button

# Number for rec

# record cover

# play button

# like button

# title

# artist

# similarity index

	# colored circle with color as measure of similarity

# embedding comparison

# compare spotify metrics with query

	# three columns - query vs rec with diff in middle (colored circle)

## must be scrollable div inside fixed 100% div with x close in corner or back




########################## Umap Plot Components ##########################

# plot

# controls

########################## Audio Player Components ##########################

# play button

# pause button

# next button

# back button

# add playlist to Spotify



########################## Body ##########################


sonufy_nav = html.Div(id='sonufy_nav', className='sixteen open', children=[
	html.Div(id='nav_hover', className='hover'),
	*sonufy_block,
	html.Div(id='close_nav', className='close no_display')
	]
	)

this_song = html.Div(id='this_song', className='sixteen', children=[
	html.Div(id='this_hover', className='hover'),
	html.Div(id='this_song_div', className='track_div', children=[
		html.H1('this song', id='this_title'),
	]),
	html.Div(id='close_this', className='close')
])

recs = [html.Div(id=f'rec_{i}', className='sixteen recs', children=[
	html.Div(id=f'rec_{i}_hover', className='hover'),
	html.Div(id=f'rec_div_{i}', className='track_div', children=[
		html.H1(f'{i}'),
	]),
	html.Div(id=f'close_rec_{i}', className='close')
	]) for i in range(1,11)]


umap_plot = html.Div(id='umap_plot', className='umap_plot', children=[
	html.H1('Umap Plot')
	])

# audio_player = html.Div(id='radio', className='radio', children=[])

body = [sonufy_nav, this_song]
body.extend(recs)
body.append(umap_plot)


########################## Layout ##########################



app.layout = html.Div(id='main', children=[
	html.Div(className='body', children=body),
	visdcc.Run_js(id = 'toggle_open'),

])

@app.callback(
	Output('toggle_open', 'run'),
	Input('main', 'n_clicks')
	)
def toggle_open(n_clicks):
	if n_clicks == 1:
		return """
    $('.hover').click(function(){
	    if ($(this).hasClass('open')){
	    }else{
	    	$(this).parent().addClass('open')
	    }
	    	
    });

    $('.close').click(function(){
	    $('.sixteen').removeClass('open')
    });

    """
	else:
		return ''



########################## Functions ##########################

@app.callback(
	Output('nav_full', 'children'),
	Output('sonufy_nav', 'className'),
	Output('close_nav', 'className'),
	Output('this_song_div', 'children'),
	*[Output(f'rec_div_{i}', 'children') for i in range(1,11)],
	Input(component_id='search_input', component_property='value')
	)
def search(search_input):

	if search_input is None:
		raise PreventUpdate

	try:
		track, latents, this_track, audio_features = sonufy.search_for_recommendations(search_input, get_time_and_freq=True)
		
		features_to_use = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
		

		this_song_layout = [
			html.Div(id='this_header_div', className='this_header_div', children=[
					
				# number
				html.H1('this song', className='artist_info'),

				html.Img(className='cover_art', src=track['album']['images'][-1]['url']),

				# title and artist
				html.P(className='artist_info', children=[
					html.Span(track['name'], className='track_name'), 
					' by ',
					html.Span(track['artists'][0]['name'], className='artist')
					]),

				# cover


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
		]

		recs_layout = []

		for i in range(1,11):

			this_rec_layout = [

				
				html.Div(id=f'rec_header_div_{i}', className='rec_header_div', children=[
					
				# 	# number
					html.H1(i),

					html.Img(className='cover_art', src=latents.loc[i-1, 'album_art_url']),

				# 	# title and artist
					html.P(className='artist_info', children=[html.Span(latents.loc[i-1 ,'track_name']), ' by ',html.Span(latents.loc[i-1 ,'artist_name'], className='artist')]),



					]),

				html.Div(id=f'rec_full_{i}', className='rec_full', children=[
					html.Div(id=f'rec_player_{i}', className='rec_player', children=[
						# play button
						# like button
						]),
					
					html.Div(id=f'rec_info_{i}', className='rec_info', children=[
						# vector similarity
						html.Div(id=f'rec_vector_compare_{i}', className='rec_vector_compare', children=[
							html.Div(className='rec_vector vector', children=[
								html.Div(str(round(col, 2)), className='vector_cell', style={'background':f'rgba({100*col},0,{-100*col},1)'}) for col in latents.loc[i-1, sonufy.latent_cols]
								]),

							html.Div(className='query_vector vector', children=[
								html.Div(str(round(col, 2)), className='vector_cell', style={'background':f'rgba({100*col},0,{-100*col},1)'}) for col in list(*this_track.values)
								])
								
							]),
						# similarity indexes
						html.Div(id=f'rec_similarity_{i}', className='rec_similarity', children=[
							html.Div(className='total_similarity similarity_index', children=[
								'similarity: ' + str(latents.loc[i-1, 'similarity'])
								]),
							html.Div(className='time_similarity similarity_index', children=[
								'time similarity: ' + str(latents.loc[i-1, 'time_similarity'])
								]),
							html.Div(className='freq_similarity similarity_index', children=[
								'frequency similarity: ' + str(latents.loc[i-1, 'frequency_similarity'])
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
										audio_features[0][feature]
										]),

									html.Div(className='diff_feature', children=[
										html.Div(feature, className='diff_title'),
										html.Div(round(abs(audio_features[0][feature] - audio_features[i][feature]),2), className='diff_quant')
										]),

									html.Div(className='rec_feature', children=[
										audio_features[i][feature]
										]),
									]) for feature in features_to_use]
							])

						])

					])
			]
			recs_layout.append(this_rec_layout)

		umap_layout = []

		return track['name'], 'sixteen', 'close', this_song_layout, *recs_layout


	except:
		print('error')
		pass


########################## Run ##########################
if __name__ == "__main__":
    debug = config.debug
    app.run_server(debug=debug, host=config.host, port=config.port)