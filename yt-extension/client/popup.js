document.getElementById("activate").addEventListener("click", removeComments);

function removeComments() {
    // Send a message to the content script to remove comments
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        chrome.tabs.sendMessage(tabs[0].id, {action: "removeComments"});
    });
}