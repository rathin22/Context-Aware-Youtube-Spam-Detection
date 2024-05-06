async function fetchSpamStatuses(commentTexts, _videoId) {
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json',},
        body: JSON.stringify({
            videoId: _videoId,
            comments: commentTexts
        })
    });
    const data = await response.json();

    return data.spam
}

async function hideSpamComments(videoId) {
    const commentElements = document.querySelectorAll('ytd-comment-thread-renderer');
    const commentTexts = Array.from(commentElements).map(
        commentElement => commentElement.querySelector('#content-text').innerText
    );

    const spamStatuses = await fetchSpamStatuses(commentTexts, videoId);
    if(spamStatuses) {
        commentElements.forEach((element, index) => {
            hideCommentIfSpam(element, spamStatuses[index])
            // highlightComment(element, spamStatuses[index])
        });
    }

}

function hideCommentIfSpam(element, isSpam){
    if(isSpam){
        element.style.display = 'none';
    }
}
// Used for demonstration and testing purposes
function highlightComment(element, isSpam){
    if (isSpam) {
        element.style.backgroundColor = 'red';
        //element.style.color = 'white'; // Change text color for better readability
    }
    else {
        element.style.backgroundColor = 'lightgreen';
    }
}


async function fetchVideoDetails(videoId) {
    const apiKey = 'AIzaSyB6qaMwoz9K9QXYE4es087YTiZFhbngyKo';
    const url = `https://www.googleapis.com/youtube/v3/videos?id=${videoId}&key=${apiKey}&part=snippet`;

    const response = await fetch(url);
    const data = await response.json();
    if (data.items.length > 0) {
        const snippet = data.items[0].snippet;
        console.log("API response")
        let tags = ""
        if ("tags" in snippet){
            tags = snippet.tags
        }
        return {
            title: snippet.title,
            description: snippet.description,
            tags: tags,
            category: snippet.categoryId
        };
    }

    return {};
}

function getVideoIdFromUrl() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('v');  // 'v' is the parameter name for video IDs in YouTube URLs
}

async function sendVideoDetails(videoId) {
    const videoInfo = await fetchVideoDetails(videoId)
    console.log(videoInfo)

    // Send context information
    await fetch('http://localhost:5000/send_video_details', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({...videoInfo, videoId})
    });
}

async function initializeSpamFiltering() {
    const videoId = getVideoIdFromUrl();
    if (!videoId) {
        console.error('No video ID found');
        return;
    }
    
    sendVideoDetails(videoId)

    hideSpamComments(videoId)
}



// initializeSpamFiltering()
setInterval(initializeSpamFiltering, 10000)