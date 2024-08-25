// Home.js

import React, { useState } from 'react';
import './Home.css';

function Home() {
    // Define state for prediction result
    const [predictionResult, setPredictionResult] = useState("");

    // JavaScript functions here
    const performPrediction = () => {
        // Perform prediction logic here
        console.log("Performing prediction...");
        // Example: Set prediction result to some value
        setPredictionResult("Prediction performed successfully!");
    }

    const displayResults = () => {
        // Display prediction results here
        console.log("Prediction result: " + predictionResult);
    }

    return (
        <div className="container">
            <div className="header">
                <div className="title">Oral Cancer Prediction</div>
                <div className="links">
                    <a href="/">Home</a>
                    <a href="/signin">Sign-In</a>
                    <a href="/signup">Sign-Up</a>
                </div>
            </div>
            <div className="main">
                <h2>Oral Cancer Prediction using CNN and RNN</h2>
        
            </div>
        </div>
    );
}

export default Home;
