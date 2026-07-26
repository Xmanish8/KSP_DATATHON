const http = require('http');

const { API_HOST, API_PORT, connectionError } = require('../utils/config');

module.exports = function(districtA, districtB) {
  const url = `http://${API_HOST}:${API_PORT}/compare?districts=${encodeURIComponent(districtA)},${encodeURIComponent(districtB)}`;

  http.get(url, (res) => {
    let raw = '';
    res.on('data', chunk => raw += chunk);
    res.on('end', () => {
      try {
        const json = JSON.parse(raw);
        if (json.status !== 'ok') {
          console.error(`❌ Error: ${json.message}`);
          process.exitCode = 1;
          return;
        }
        const comp = json.data.comparison;
        console.log('\n============================================================');
        console.log(`🛡️  SurakshaAI Side-by-Side District Comparison`);
        console.log('============================================================');
        comp.forEach(d => {
          console.log(`\n📍 ${d.district} (Year ${d.year})`);
          console.log(`   Label: ${d.label} | Total Score: ${d.total_score} / 100`);
          console.log(`   D1 Violent: ${d.dimensions.D1_violent_crime} | D2 Property: ${d.dimensions.D2_property_crime} | D3 Women: ${d.dimensions.D3_women_safety}`);
        });
        console.log('\n============================================================\n');
      } catch (err) {
        console.error('❌ Failed to parse API response:', err.message);
        process.exitCode = 1;
      }
    });
  }).on('error', (e) => {
    connectionError(API_HOST, API_PORT);
  });
};