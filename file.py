def loginToSpotify():
    with open(r"credentials.txt") as f:
        [SPOTIPY_CLIENT_ID,SPOTIPY_CLIENT_SECRET] = f.read().split("\n")
        f.close()
    auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

def searchPlaylist(sp):
    playlistLink = input("Please Enter Playlist URL\n")
    playlistDict = sp.playlist(playlistLink)
    return playlistDict



## SP is the api connection (spotify object)
sp = loginToSpotify()
playlistDict = searchPlaylist(sp)

totalSongs = playlistDict['tracks']['total']
song_list = []
song_id = []
artist_id = []

playlistSongsFile = open('playlistSongs.csv', 'w')

for i in range(totalSongs):
    artists = [k["name"] for k in playlistDict['tracks']['items'][i]["track"]["artists"]]
    trackName = playlistDict['tracks']['items'][i]['track']['name']
    artistName = artists[0]
    songID = playlistDict['tracks']['items'][i]['track']['id']
    uri = playlistDict['tracks']['items'][i]['track']['uri']

    print(f"{trackName}, {artistName}, {songID}, {uri}")
    
    print(sp.audio_features(songID)[0])