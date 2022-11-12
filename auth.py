from flask import Flask, request, redirect, render_template, url_for, session



server = Flask(__name__)
server.secret_key = 'effro'

# Spotify URLS
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE_URL = "https://api.spotify.com"
API_VERSION = "v1"
SPOTIFY_API_URL = "{}/{}".format(SPOTIFY_API_BASE_URL, API_VERSION)

SCOPE = "user-top-read user-follow-modify user-library-read playlist-read-private " \
        "playlist-read-collaborative user-follow-read user-library-modify"
STATE = ""
SHOW_DIALOG_bool = True
SHOW_DIALOG_str = str(SHOW_DIALOG_bool).lower()


class SpotifyAPI():
    access_token = None
    refresh_token = None
    token_type = None
    expires_in = None
    access_token_expires = datetime.datetime.now()
    access_token_did_expire = True
    authorization_header = None

    def get_auth_query_parameters(self):
        return {
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPE,
            # "state": STATE,
            # "show_dialog": SHOW_DIALOG_str,
            "client_id": CLIENT_ID
        }

    def get_access_token_data(self, auth_token):
        return {
            "grant_type": "authorization_code",
            "code": str(auth_token),
            "redirect_uri": REDIRECT_URI,
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
        }

    @route("/")
    def index(self):
        """
        Create authorization url and redirect to it
        """
        url_args = "&".join(["{}={}".format(key, quote(val)) for key, val in self.get_auth_query_parameters().items()])
        auth_url = "{}/?{}".format(SPOTIFY_AUTH_URL, url_args)
        return redirect(auth_url)

    @route("/callback/q")
    def perform_auth(self):
        # Requests refresh and access tokens
        auth_token = request.args['code']  # access the data from the GET (url)
        access_token_data = self.get_access_token_data(auth_token)
        post_request = requests.post(SPOTIFY_TOKEN_URL, data=access_token_data)
        if post_request.status_code not in range(200, 299):
            self.perform_auth()
        # Tokens are Returned to Application
        response_data = json.loads(post_request.text)
        self.access_token = response_data["access_token"]
        self.refresh_token = response_data["refresh_token"]
        self.token_type = response_data["token_type"]
        self.expires_in = response_data["expires_in"]
        now = datetime.datetime.now()
        expires = now + datetime.timedelta(seconds=self.expires_in)
        self.access_token_expires = expires
        self.access_token_did_expire = expires < now
        return self.callback()

    def get_persistent_access_token(self):
        token = self.access_token
        expires = self.access_token_expires
        now = datetime.datetime.now()
        print(f'Attemp to access server. Token expires at {expires}')
        if expires < now:
            self.perform_auth()
            return self.get_persistent_access_token()
        elif token is None:
            self.perform_auth()
            return self.get_persistent_access_token()
        return token

    def set_authorization_header(self):
        access_token = self.get_persistent_access_token()
        self.authorization_header = {"Authorization": "Bearer {}".format(access_token)}
        return True

    def get_profile_data(self):
        endpoint = f"{SPOTIFY_API_URL}/me"
        r = requests.get(endpoint, headers=self.authorization_header)
        return json.loads(r.text)

    def get_user_playlist_data(self, limit=50, offset=0):
        endpoint = f"{SPOTIFY_API_URL}/me/playlists?limit={limit}&offset={offset}"
        r = requests.get(endpoint, headers=self.authorization_header)
        return json.loads(r.text)

    def get_user_top_artists_and_tracks(self, entity_type="artists", limit=50,
                                        time_range="medium_term", offset=0):
        """
        :param entity_type:  artists or tracks
        :param limit: The number of entities to return. Minimum: 1. Maximum: 50.
        :param time_range: Over what time frame the affinities are computed. Valid values: long_term
            (calculated from several years of data and including all new data as it becomes available),
            medium_term (approximately last 6 months), short_term (approximately last 4 weeks)
        :param offset: The index of the first entity to return. Default: 0 (i.e., the first track).
            Use with limit to get the next set of entities
        :return: json text
        """
        endpoint = f"{SPOTIFY_API_URL}/me/top/{entity_type}?time_range={time_range}&limit={limit}" \
                   f"&offset={offset}"
        r = requests.get(endpoint, headers=self.authorization_header)
        return json.loads(r.text)

    def get_user_followed_artists(self, entity_type="artist", limit=50):
        endpoint = f"{SPOTIFY_API_URL}/me/following/?type={entity_type}&limit={limit}"
        r = requests.get(endpoint, headers=self.authorization_header)
        return json.loads(r.text)

    def get_user_saved_albums(self, limit=50, offset=0):
        endpoint = f"{SPOTIFY_API_URL}/me/albums/?limit={limit}&offset={offset}"
        r = requests.get(endpoint, headers=self.authorization_header)
        return json.loads(r.text)

    def get_user_saved_tracks(self, limit=50, offset=0):
        endpoint = f"{SPOTIFY_API_URL}/me/tracks/?limit={limit}&offset={offset}"
        r = requests.get(endpoint, headers=self.authorization_header)
        return json.loads(r.text)

    def get_all_user_top_artists_and_tracks(self, entity_type="artists", time_range="medium_term"):
        """
        Return a DataFrame containing all of user's top artists or tracks
        :return: pandas DataFrame
        """
        total_top_entity = self.get_user_top_artists_and_tracks(entity_type=entity_type, limit=1,
                                                                time_range=time_range, offset=0)['total']
        user_top_entity_data = pd.DataFrame()
        for i in range(int(total_top_entity/50)+1):
            temp_json = self.get_user_top_artists_and_tracks(entity_type=entity_type, limit=50,
                                                             time_range=time_range, offset=i*50)
            if entity_type == "artists":
                temp = process_user_top_artists(temp_json)
            else:
                temp = process_user_top_tracks(temp_json)
            user_top_entity_data = pd.concat([user_top_entity_data, temp])
        return user_top_entity_data

    def get_related_artists(self, artist_id):
        """
        :param artist_id: the artist's id provided by the user_top_artists_data_long_term 50 artists
        :return: json text
        """
        endpoint = f"{SPOTIFY_API_URL}/artists/{artist_id}/related-artists"
        r = requests.get(endpoint, headers=self.authorization_header)
        return json.loads(r.text)

    def get_all_user_saved_tracks(self):
        """
        Return a DataFrame containing all of user's saved tracks
        :return: pandas DataFrame
        """
        total_songs_saved = self.get_user_saved_tracks(limit=1, offset=0)['total']
        user_saved_tracks_data = pd.DataFrame()
        for i in range(int(total_songs_saved/50)+1):
            temp_json = self.get_user_saved_tracks(limit=50, offset=i*50)
            temp = process_user_saved_tracks_data(temp_json)
            user_saved_tracks_data = pd.concat([user_saved_tracks_data, temp])
        return user_saved_tracks_data

    def get_audio_features(self, track_ids):
        """
        :param track_ids: spotify track ids with comma as url character (%2C) between them
        :return: json text
        """
        endpoint = f"{SPOTIFY_API_URL}/audio-features?ids={track_ids}"
        r = requests.get(endpoint, headers=self.authorization_header)
        return json.loads(r.text)

    def callback(self):
        self.set_authorization_header()

        user_profile_data = self.get_profile_data()

        user_saved_tracks_data = self.get_all_user_saved_tracks()

        user_top_artists_data_medium_term = \
            self.get_all_user_top_artists_and_tracks(entity_type="artists", time_range="medium_term")
        user_top_artists_data_long_term = \
            self.get_all_user_top_artists_and_tracks(entity_type="artists", time_range="long_term")
        user_top_artists_data_short_term = \
            self.get_all_user_top_artists_and_tracks(entity_type="artists", time_range="short_term")

        related_artists_total = pd.DataFrame()
        for i, x in enumerate(user_top_artists_data_long_term['external_url'].head(10)):
            print(f'Getting related artists {i}')
            related_artists_json = self.get_related_artists(x.rsplit('/', 1)[-1])
            related_artists = process_related_artists(related_artists_json)
            related_artists_total = pd.concat([related_artists_total, related_artists])
        related_artists_total = related_artists_total.drop_duplicates(subset=['external_url', 'name'])

        user_top_tracks_data_medium_term = \
            self.get_all_user_top_artists_and_tracks(entity_type="tracks", time_range="medium_term")
        user_top_tracks_data_long_term = \
            self.get_all_user_top_artists_and_tracks(entity_type="tracks", time_range="long_term")
        user_top_tracks_data_short_term = \
            self.get_all_user_top_artists_and_tracks(entity_type="tracks", time_range="short_term")

        user_followed_artists_data_data = self.get_user_followed_artists()
        user_followed_artists_data_data = process_user_followed_artists_data(user_followed_artists_data_data)

        path = f"../data/{user_profile_data['id']}/"
        if not os.path.exists(path):
            os.makedirs(path)

        users = pd.DataFrame(columns=['username', 'id', 'href', 'followers', 'accessed_at'])
        users.loc[-1] = [user_profile_data['display_name'], user_profile_data['id'],
                         user_profile_data['href'], user_profile_data['followers']['total'],
                         datetime.datetime.now()]
        session['user_id'] = user_profile_data['id']
        if not os.path.isfile('../data/users.csv'):
            users.to_csv('../data/users.csv', index=False)
        else:
            users.to_csv('../data/users.csv', index=False, mode='a', header=False)

        user_saved_tracks_data.to_csv(f"{path}user_saved_tracks_data.csv", index=False)

        user_top_artists_data_medium_term.to_csv(f"{path}user_top_artists_data_medium_term.csv", index=False)
        user_top_tracks_data_medium_term.to_csv(f"{path}user_top_tracks_data_medium_term.csv", index=False)

        user_top_artists_data_long_term.to_csv(f"{path}user_top_artists_data_long_term.csv", index=False)
        user_top_tracks_data_long_term.to_csv(f"{path}user_top_tracks_data_long_term.csv", index=False)

        user_top_artists_data_short_term.to_csv(f"{path}user_top_artists_data_short_term.csv", index=False)
        user_top_tracks_data_short_term.to_csv(f"{path}user_top_tracks_data_short_term.csv", index=False)

        user_followed_artists_data_data.to_csv(f"{path}user_followed_artists_data_data.csv", index=False)

        user_top_tracks_all_periods = pd.concat([user_top_tracks_data_medium_term, user_top_tracks_data_long_term,
                                                 user_top_tracks_data_short_term, user_saved_tracks_data])
        user_top_tracks_all_periods = user_top_tracks_all_periods.drop_duplicates(subset=['song_external_url'])
        track_ids = user_top_tracks_all_periods['song_external_url']

        start = 0
        step = 50
        user_tracks_features = pd.DataFrame()
        for i in range(0, track_ids.shape[0], step):
            print(f'Getting audio features {int(start/step)}')
            track_ids_temp = '%2C'.join(track_ids[start:start+step].apply(lambda t: t.rsplit('/', 1)[-1]).tolist())
            user_tracks_features_json = self.get_audio_features(track_ids_temp)
            user_tracks_features_temp = process_audio_features(user_tracks_features_json)
            user_tracks_features = pd.concat([user_tracks_features, user_tracks_features_temp])
            start += step

        user_tracks_features.to_csv(f"{path}user_tracks_audio_features.csv", index=False)
        related_artists_total.to_csv(f"{path}related_artists.csv", index=False)

        return run_dash(app)
        # return redirect(url_for("/app/"))
        # return render_template("index.html", sorted_array=user_tracks_features)