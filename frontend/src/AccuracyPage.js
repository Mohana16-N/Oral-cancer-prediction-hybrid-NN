// import React from 'react';
// import image1 from 'C:\Users\theja\OneDrive\Desktop\FIRSTAPP\frontend\image1.jpg'; // Import the image files
// import image2 from 'C:\Users\theja\OneDrive\Desktop\FIRSTAPP\frontend\image2.jpg';
// import image3 from 'C:\Users\theja\OneDrive\Desktop\FIRSTAPP\frontend\image3.jpg';
// import image4 from 'C:\Users\theja\OneDrive\Desktop\FIRSTAPP\frontend\image4.jpg';
// import image5 from 'C:\Users\theja\OneDrive\Desktop\FIRSTAPP\frontend\image5.jpg';
// import './AccuracyPage.css'; // Import the CSS file
import './AccuracyPage.css'; // Import the CSS file
function AccuracyPage() {
    return (
        <div className="image">
                <img src={require("./image5.jpg")} />
                <img src={require("./image3.jpg")} />
                <img src={require("./image4.jpg")} />
                <img src={require("./image1.jpg")} />
                <img src={require("./image2.jpg")} />
        </div>
        
    );
}

export default AccuracyPage;
