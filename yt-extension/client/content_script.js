async function fetchSpamStatuses(commentTexts) {
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({comments: commentTexts})
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

async function highlightSpamComments() {
    const commentElements = document.querySelectorAll('ytd-comment-thread-renderer'); // Corrected variable name
    const commentTexts = Array.from(commentElements).map(commentElement => commentElement.querySelector('#content-text').innerText);

    const spamStatuses = await fetchSpamStatuses(commentTexts);
    commentElements.forEach((element, index) => {
        if (spamStatuses[index]) {
            console.log(element)
            console.log(spamStatuses[index])
            console.log("")
            element.style.backgroundColor = 'red';
            element.style.color = 'white'; // Change text color for better readability
        }
        else {
            element.style.backgroundColor = 'green';
        }
    });
}

// Run this function periodically to handle dynamic comment loading
setInterval(highlightSpamComments, 10000);