import React from 'react';
import { Routes, Route } from 'react-router-dom';
import ImageUploadPage from './ImageUploadPage';
import Login from './Login';
import Signup from './Signup';
import Home from './Home';
import AccuracyPage from './AccuracyPage';

function App() {
    return (
        <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/signin" element={<Login />} />
            <Route path="/signup" element={<Signup />} />
            <Route path="/accuracy" element={<AccuracyPage />} />
            <Route path="/home" element={<ImageUploadPage />} />
        </Routes>
    );
}

export default App;
