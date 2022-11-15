
window.onSpotifyWebPlaybackSDKReady = () => {

     $('.hover').click(function(){
        if ($(this).hasClass('open')){
        }else{
            $(this).parent().addClass('open')
        }
                
    });

    $('.close').click(function(){
        $('.sixteen').removeClass('open')
    });


    const token_string = document.cookie;
	let token = token_string.split('access_token=')
	token = token[token.length-1]

    const player = new Spotify.Player({
        name: 'Sonufy Player',
        getOAuthToken: cb => { cb(token); },
        volume: 1
    });

    // Ready
    player.addListener('ready', ({ device_id }) => {
        console.log('Ready with Device ID', device_id);
    });

    // Not Ready
    player.addListener('not_ready', ({ device_id }) => {
        console.log('Device ID has gone offline', device_id);
    });

    player.addListener('initialization_error', ({ message }) => {
        console.error(message);
    });

    player.addListener('authentication_error', ({ message }) => {
        console.error(message);
    });

    player.addListener('account_error', ({ message }) => {
        console.error(message);
    });

    
    player.connect();

}
