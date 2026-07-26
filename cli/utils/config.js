const API_HOST = process.env.API_HOST || '127.0.0.1';
const API_PORT = process.env.API_PORT || 5000;

function connectionError(host, port) {
  console.error(`❌ Could not connect to SurakshaAI API server at http://${host}:${port}`);
  console.error('   Ensure Flask API is running (`python api/app.py`)');
  process.exitCode = 1;
}

module.exports = { API_HOST, API_PORT, connectionError };