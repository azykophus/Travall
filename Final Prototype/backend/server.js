const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const port = 3002; // Ensure this port matches the one your frontend is addressing
const { spawn } = require('child_process');

app.use(cors()); // Enable CORS
app.use(bodyParser.json()); // Middleware to parse JSON bodies

// GET endpoint to provide initial data
app.get('/message', (req, res) => {
  // Mock database object for demonstration
  const data = {
    searchQuery: "Default Query",
    city: "Default City",
    accessibilityOption: "None"
  };
  res.json(data);
});


app.post('/submit', (req, res) => {
  const { searchQuery, city, accessibilityOption } = req.body;
  
  const wheelchairAccess = accessibilityOption.includes('wheelchair');
  const visualAccess = accessibilityOption.includes('visually-impaired');

  const pythonProcess = spawn('python3', ['script.py', city, searchQuery, wheelchairAccess.toString(), visualAccess.toString()]);


  let outputData = '';
  pythonProcess.stdout.on('data', (data) => {
    outputData += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Error from Python script: ${data}`);
  });
  pythonProcess.on('close', (code) => {
    console.log(`Python script process exited with code ${code}`);
    // Respond with the output received from the Python script
   
      const parsedData = JSON.parse(outputData);
      console.log(parsedData)
      // res.json({hi:"hi"});
      res.json({ message: parsedData });
      
    
  });
});


// Start the server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});




// const express = require('express');
// const bodyParser = require('body-parser');
// const cors = require('cors');
// const { spawn } = require('child_process');

// const app = express();
// const port = 3002;

// app.use(cors());
// app.use(bodyParser.json());

// app.post('/submit', (req, res) => {
//   const { searchQuery, city, accessibilityOption } = req.body;

//   // Execute the Python script
//   const pythonProcess = spawn('python', ['script.py', city, searchQuery]);

//   // Collect data from the Python script
//   let outputData = '';
//   pythonProcess.stdout.on('data', (data) => {
//     outputData += data.toString();
//   });

//   pythonProcess.stderr.on('data', (data) => {
//     console.error(`Error from Python script: ${data}`);
//   });

//   pythonProcess.on('close', (code) => {
//     console.log(`Python script process exited with code ${code}`);
//     // Respond with the output received from the Python script
//     res.json({ message: outputData });
//   });
// });

// app.listen(port, () => {
//   console.log(`Server running on http://localhost:${port}`);
// });
