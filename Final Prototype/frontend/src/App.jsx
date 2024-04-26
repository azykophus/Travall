import React, { useState } from 'react';
import './App.css'; // This will be our stylesheet
// import axios from 'axios';
import AuthPage from "./AuthPage"
import ChatsPage from "./ChatsPage"

function App() {
  const [user,setUser] = useState(undefined)
  if(!user){
    return <AuthPage onAuth = {(user)=>setUser(user)}/>
  }else{
    return <ChatsPage user = {user}/>;
  }
  
  // return (
  //   <div className="App">
  //     <header className="App-header">
  //       <nav className="App-nav">
  //         <div className="logo">Travall</div>
  //         <ul>
  //           <li>Home</li>
  //           <li>About</li>
  //           <li>Destinations</li>
  //           <li>GitHub</li>
  //           <li>Contact</li>
  //         </ul>
  //       </nav>
  //       <div className="hero">
  //         <h1>Where do you want to go?</h1>
  //         <div className="search-bar">
  //           <input type="text" placeholder="Search your destination..." onChange={e => setSearchQuery(e.target.value)} />
  //           <select className="city-options" onChange={e => setCity(e.target.value)}>
  //             <option value="">City</option>
  //             <option value="INDIA">INDIA</option>
  //             <option value="delhi">Delhi</option>
  //             <option value="agra">Agra</option>
  //             <option value="bengaluru">Bengaluru</option>
  //             <option value="chennai">Chennai</option>
  //             <option value="hyderabad">Hyderabad</option>
  //             <option value="jaipur">Jaipur</option>
  //             <option value="kolkata">Kolkata</option>
  //             <option value="mumbai">Mumbai</option>
  //             <option value="pune">Pune</option>
  //             <option value="udaipur">Udaipur</option>
  //           </select>
  //           <select className="accessibility-options" onChange={e => setAccessibilityOption(e.target.value)}>
  //             <option value="">Accessibility Options </option>
  //             <option value="wheelchair">Wheelchair Accessible</option>
  //             <option value="visually-impaired">Visually Impaired</option>
  //           </select>
  //           <button className="search-button" onClick={handleSubmit}>
  //             <img src="/icon.png" alt="Search" />
  //           </button>
  //         </div>
          
  //         <div className="toggle-container">
  //           <h2><span className="option" id="option1">Our Model</span></h2>
  //           <label className="toggle">
  //             <input type="checkbox" id="toggle-switch" />
  //             <span className="slider round"></span>
  //           </label>
            
  //           <span className="option" id="option2">RAG</span>
  //         </div>

  //         <div className="output-container">
  //           <p>{output}</p>
  //         </div>
  //       </div>
  //     </header>
  //   </div>
  // );
}

export default App;