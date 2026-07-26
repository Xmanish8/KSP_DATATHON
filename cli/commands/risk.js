const http = require('http');

const { API_HOST, API_PORT, connectionError } = require('../utils/config');

module.exports = function(district, options) {
  const yearQuery = options.year ? `?year=${options.year}` : '';
  const url = `http://${API_HOST}:${API_PORT}/risk/${encodeURIComponent(district)}${yearQuery}`;

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
        const data = json.data;
        console.log('\n============================================================');
        console.log(`🛡️  SurakshaAI Risk Score: ${data.district.toUpperCase()} (${data.year})`);
        console.log('============================================================');
        console.log(`   RISK LABEL : [ ${data.label} ]`);
        console.log(`   TOTAL SCORE: ${data.total_score} / 100`);
        console.log('\n   6-Dimension Score Breakdown:');
        console.log(`    • Violent Crime Severity (D1) : ${data.dimensions.D1_violent_crime} / 20`);
        console.log(`    • Property & Fraud Crime (D2) : ${data.dimensions.D2_property_crime} / 20`);
        console.log(`    • Women's Safety (D3)        : ${data.dimensions.D3_women_safety} / 15`);
        console.log(`    • Trend Volatility (D4)      : ${data.dimensions.D4_trend_volatility} / 15`);
        console.log(`    • Total Volume Index (D5)    : ${data.dimensions.D5_total_volume} / 15`);
        console.log(`    • Anomaly/Spike Flag (D6)    : ${data.dimensions.D6_anomaly_spike} / 15`);
        console.log('============================================================\n');
      } catch (err) {
        console.error('❌ Failed to parse API response:', err.message);
        process.exitCode = 1;
      }
    });
  }).on('error', (e) => {
    connectionError(API_HOST, API_PORT);
  });
};