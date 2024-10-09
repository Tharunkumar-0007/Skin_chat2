document.addEventListener('DOMContentLoaded', function() {
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query');
    const chatMessages = document.getElementById('chat-messages');
    const voiceButton = document.getElementById('voice-input');

    // Voice recognition setup
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';

    // Handle voice input
    voiceButton.addEventListener('click', () => {
        recognition.start();
    });

    recognition.onresult = (event) => {
        queryInput.value = event.results[0][0].transcript;

        // Automatically click the send button after 2 seconds
        setTimeout(() => {
            queryForm.dispatchEvent(new Event('submit'));
        }, 2000);
    };

    // Handle form submission (text and voice input)
    queryForm.onsubmit = async function(e) {
        e.preventDefault();
        const query = queryInput.value;

        // Append user message to the chat
        appendMessage(query, 'user-message');
        queryInput.value = '';

        // Send the query to the backend
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `query=${encodeURIComponent(query)}`,
        });

        const result = await response.json();

        // Append bot response to the chat
        appendMessage(result.response, 'bot-message');
    };

    // Function to append message to the chat window
    function appendMessage(text, className) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', className);
        messageDiv.textContent = text;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll to the bottom
    }
});
