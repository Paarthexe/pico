const { defineConfig } = require('@playwright/test');
module.exports = defineConfig({
  testDir: './tests/browser',
  timeout: 60000,
  expect: { timeout: 15000 },
  workers: 1,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: { baseURL: 'http://127.0.0.1:8501', trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  webServer: {
    command: `${process.env.PICO_PYTHON || 'python3'} -m streamlit run app.py --server.headless=true --server.address=127.0.0.1 --server.port=8501`,
    url: 'http://127.0.0.1:8501/_stcore/health',
    reuseExistingServer: !process.env.CI,
    timeout: 60000,
  },
});
