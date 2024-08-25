import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate, Link } from 'react-router-dom';
import './ImageUploadPage.css';

function ImageUploadPage() {
    const [selectedImage, setSelectedImage] = useState(null);
    const [imageUrl, setImageUrl] = useState(null);
    const [result, setResult] = useState(null);
    const navigate = useNavigate();

    const handleImageChange = (event) => {
        setSelectedImage(event.target.files[0]);
    }

    const handleSubmit = (event) => {
        event.preventDefault();
        if (selectedImage) {
            const formData = new FormData();
            formData.append('file', selectedImage);  // Use 'file' instead of 'image'
            
            axios.post('http://127.0.0.1:5000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'  // Set content type as multipart/form-data
                }
            })
            .then(response => {
                console.log('Response from backend:', response.data);
                setResult(response.data.result);
                setImageUrl(URL.createObjectURL(selectedImage));
            })
            .catch(error => {
                console.log('Error uploading image:', error);
                alert("Failed to upload image");
            });
        } else {
            alert("Please select an image to upload.");
        }
    }
    

    const handleLogout = () => {
        navigate('/');
    };

    return (
        <div className="container">
            <div className="header">
                <div className="title">Oral Cancer Prediction</div>
                <div className="links">
                    <button className="logout" onClick={handleLogout}>Logout</button>
                    <Link to="/accuracy" className="accuracy-button">Accuracy</Link>
                </div>
            </div>
            <div className="main-content">
                <h2 className="upload-heading">Upload Image</h2>
                <form onSubmit={handleSubmit} className="form-group">
                    <input
                        type="file"
                        accept="image/png, image/jpeg, image/jpg"
                        onChange={handleImageChange}
                        className="form-control-file"
                    />
                    <button type="submit" className="btn btn-primary">Upload</button>
                </form>
                {imageUrl && (
                    <div className="preview">
                        <h5>Preview:</h5>
                        <img
                            src={imageUrl}
                            alt="Uploaded"
                            className="preview-image"
                        />
                    </div>
                )}
                {result !== null && (
                    <div className="result">
                        <h5>Prediction Result:</h5>
                        <p>{result}</p>
                    </div>
                )}
            </div>
            <div className="mouth1">
                {/* <img src={require("./mouth1.jpg")} alt="Mouth" /> */}
            </div>
        </div>
    );
}

export default ImageUploadPage;
