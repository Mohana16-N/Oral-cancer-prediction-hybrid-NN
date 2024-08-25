const express = require('express');
const mysql = require('mysql');
const cors = require('cors');

const app = express();
app.use(express.json());
app.use(cors());

// MySQL database connection
const db = mysql.createConnection({
  host: "localhost",
  user: "root",
  password: "",
  database: "signup"
});

// Connect to the database
db.connect(err => {
  if (err) {
    console.error('Database connection failed:', err.stack);
    return;
  }
  console.log('Connected to database.');
});

// Signup route
app.post('/signup', (req, res) => {
  const sql = "INSERT INTO login (firstName, lastName, email, password) VALUES (?, ?, ?, ?)";
  const values = [
    req.body.firstName,
    req.body.lastName,
    req.body.email,
    req.body.password
  ];

  db.query(sql, values, (err, data) => {
    if (err) {
      console.error('Error executing query:', err);
      return res.status(500).json({ error: "Database query error" });
    }
    return res.status(201).json({ message: "User registered successfully", data });
  });
});

// Login route
app.post('/login', (req, res) => {
  const sql = "SELECT * FROM login WHERE `email` = ? AND `password` = ?";

  db.query(sql, [req.body.email, req.body.password], (err, data) => {
    if (err) {
      console.error('Error executing query:', err);
      return res.status(500).json({ error: "Database query error" });
    }
    if (data.length > 0) {
      return res.json("success");
    } else {
      return res.json("fail");
    }
  });
});

// Start the server
app.listen(8081, () => {
  console.log("Server is listening on port 8081");
});
