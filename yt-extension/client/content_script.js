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

// async function hideSpamComments() {
//     const comments = document.querySelectorAll('ytd-comment-thread-renderer');
//     const commentTexts = Array.from(comments).map(commentElement => commentElement.querySelector('#content-text').innerText);

//     const spamStatuses = await fetchSpamStatuses(commentTexts);
//     commentElements.forEach((element, index) => {
//         if (spamStatuses[index]) {
//             element.style.display = 'none';  // Hides the comment if it is classified as spam
//         }
//     });
// }

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
            category: snippet.categoryId  // You might need another API call to convert categoryId to a human-readable form
        };
    }

    return {};
}

function getVideoIdFromUrl() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('v');  // 'v' is the parameter name for video IDs in YouTube URLs
}

async function initializeSpamFiltering() {
    const videoId = getVideoIdFromUrl();
    if (!videoId) {
        console.error('No video ID found');
        return;
    }
    console.log(videoId)
    const video_info = await fetchVideoDetails(videoId)
    console.log(video_info)

    // Send context information
    await fetch('http://localhost:5000/send_video_details', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({...video_info, videoId})
    });

    // Run this function periodically to handle dynamic comment loading
    // setInterval(function(){ highlightSpamComments(videoId) }, 10000);
    highlightSpamComments(videoId)
}

async function highlightSpamComments(videoId) {
    const commentElements = document.querySelectorAll('ytd-comment-thread-renderer');
    const commentTexts = Array.from(commentElements).map(
        commentElement => commentElement.querySelector('#content-text').innerText
    );

    const spamStatuses = await fetchSpamStatuses(commentTexts, videoId);
    commentElements.forEach((element, index) => {
        if (spamStatuses[index]) {
            element.style.backgroundColor = 'red';
            element.style.color = 'white'; // Change text color for better readability
        }
        else {
            element.style.backgroundColor = 'green';
        }
    });
}

// initializeSpamFiltering()
setInterval(initializeSpamFiltering, 10000)