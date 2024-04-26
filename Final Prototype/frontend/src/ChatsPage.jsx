import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [city, setCity] = useState('');
  const [accessibilityOption, setAccessibilityOption] = useState('');
  const [output, setOutput] = useState('');
  const [message, setMessage] = useState('');

  // useEffect(() => {
  //   fetch("http://localhost:5174/message")
  //     .then((res) => res.json())
  //     .then((data) => setMessage(data.message));
  // }, []);
  const runPythonScript = () => {
    const dataToSend = {
      arg1: city,
      arg2: searchQuery
    };
  
    axios.post('http://localhost:3002/run-python', dataToSend)
      .then(response => {
        console.log(response.data); // Handle the response from the Python script
        setOutput(response.data.message);
      })
      .catch(error => {
        console.error('Error while running Python script:', error);
      });
  };
  const handleSubmit = () => {
    const searchData = {
      searchQuery: searchQuery,
      city: city,
      accessibilityOption: accessibilityOption
    };

    axios.post('http://localhost:3002/submit', searchData)
      .then(response => {
        setOutput(response.data.message);
      })
      .catch(error => {
        console.error('Error while sending data to the backend:', error);
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        <nav className="App-nav">
          <div className="logo">Travall</div>
          <ul>
            <li>Home</li>
            <li>About</li>
            <li>Destinations</li>
            <li>GitHub</li>
            <li>Contact</li>
          </ul>
        </nav>
        <div className="hero">
          <h1>Where do you want to go?</h1>
          <div className="search-bar">
            <input type="text" placeholder="Search your destination..." onChange={e => setSearchQuery(e.target.value)} />
            <select className="city-options" onChange={e => setCity(e.target.value)}>
              <option value="">City</option>
              <option value="INDIA">INDIA</option>
              <option value="delhi">Delhi</option>
              <option value="agra">Agra</option>
              <option value="bengaluru">Bengaluru</option>
              <option value="chennai">Chennai</option>
              <option value="hyderabad">Hyderabad</option>
              <option value="jaipur">Jaipur</option>
              <option value="kolkata">Kolkata</option>
              <option value="mumbai">Mumbai</option>
              <option value="pune">Pune</option>
              <option value="udaipur">Udaipur</option>
            </select>
            <select className="accessibility-options" onChange={e => setAccessibilityOption(e.target.value)}>
              <option value="">Accessibility Options</option>
              <option value="wheelchair">Wheelchair Accessible</option>
              <option value="visually-impaired">Visually Impaired</option>
            </select>
            <button className="search-button" onClick={handleSubmit}>
              <img src="/Users/abhijaysingh/Downloads/travall2/frontend/src/assets/img.jpeg" alt="Search" />
            </button>
          </div>
          
          <div className="toggle-container">
            <h2><span className="option" id="option1">Our Model</span></h2>
            <label className="toggle">
              <input type="checkbox" id="toggle-switch" />
              <span className="slider round"></span>
            </label>
            
            <span className="option" id="option2">RAG</span>
          </div>

          <div className="output-container">
            {/* <p>{message}</p> */}
            <p>{output}</p>
          </div>
        </div>
      </header>
    </div>
  );
}

export default App;





