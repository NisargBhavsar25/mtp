from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time

class UpdatedDDSEScraper:
    def __init__(self, headless=False):
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        self.chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        self.driver = None
        self.wait = None
    
    def start_driver(self):
        """Start Chrome driver with anti-detection measures"""
        self.driver = webdriver.Chrome(options=self.chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self.wait = WebDriverWait(self.driver, 30)
    
    def streamlit_login(self, username, password):
        """Updated login function based on debug findings"""
        print("Starting Streamlit-specific login process...")
        
        try:
            # Navigate to homepage
            self.driver.get("https://www.ddse-database.org/")
            
            # Wait for Streamlit app to fully load
            self.wait.until(EC.presence_of_element_located((By.XPATH, "//div[@data-testid='stApp']")))
            
            # Additional wait for dynamic content
            time.sleep(8)
            
            # Take screenshot for debugging
            self.driver.save_screenshot("streamlit_loaded.png")
            
            # Updated selectors based on your debug findings
            username_xpath = "//input[@type='text' and contains(@class, 'st-')]"
            password_xpath = "//input[@type='password' and contains(@class, 'st-')]"
            login_button_xpath = "//button[text()='Login']"
            
            # Find username field
            print("Looking for username field...")
            username_field = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, username_xpath))
            )
            print("Username field found!")
            
            # Find password field
            print("Looking for password field...")
            password_field = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, password_xpath))
            )
            print("Password field found!")
            
            # Clear and fill username
            username_field.clear()
            username_field.send_keys(username)
            time.sleep(1)
            
            # Clear and fill password
            password_field.clear()
            password_field.send_keys(password)
            time.sleep(1)
            
            # Find and click login button
            print("Looking for login button...")
            login_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, login_button_xpath))
            )
            print("Login button found, clicking...")
            login_button.click()
            
            # Wait for login to process
            time.sleep(10)
            
            # Verify login success
            if self._verify_streamlit_login():
                print("Login successful!")
                return True
            else:
                print("Login verification failed")
                return False
                
        except Exception as e:
            print(f"Login error: {e}")
            self.driver.save_screenshot("login_error_detailed.png")
            return False
    
    def _verify_streamlit_login(self):
        """Verify login success for Streamlit app"""
        print("Verifying login success...")
        
        # Look for elements that appear after successful login
        success_indicators = [
            # Streamlit selectbox (for plot controls)
            "//div[@data-testid='stSelectbox']",
            # Streamlit slider (for temperature control)
            "//div[@data-testid='stSlider']",
            # Plotly graphs
            "//div[contains(@class, 'plotly-graph-div')]",
            # Any select dropdown
            "//select",
            # Check if we're no longer on login page
            "//div[@data-testid='stSidebar']"
        ]
        
        for indicator_xpath in success_indicators:
            try:
                self.wait.until(
                    EC.presence_of_element_located((By.XPATH, indicator_xpath))
                )
                print(f"Login verified with: {indicator_xpath}")
                return True
            except:
                continue
        
        # Check URL change as fallback
        current_url = self.driver.current_url
        if "dashboard" in current_url.lower() or len(current_url) > 30:
            print("Login verified by URL change")
            return True
        
        return False
    
    def wait_for_interface_load(self):
        """Wait for the main data interface to load completely"""
        print("Waiting for data interface to load...")
        
        try:
            # Wait for key Streamlit components
            self.wait.until(
                EC.any_of(
                    EC.presence_of_element_located((By.XPATH, "//div[@data-testid='stSelectbox']")),
                    EC.presence_of_element_located((By.XPATH, "//div[@data-testid='stSlider']")),
                    EC.presence_of_element_located((By.XPATH, "//select"))
                )
            )
            
            # Additional wait for all components to render
            time.sleep(5)
            print("Interface loaded successfully")
            return True
            
        except Exception as e:
            print(f"Interface loading failed: {e}")
            return False
    
    def configure_streamlit_settings(self, temperature=25):
        """Configure plot settings in Streamlit interface"""
        print(f"Configuring settings for {temperature}°C...")
        
        try:
            # Wait for interface to be ready
            if not self.wait_for_interface_load():
                return False
            
            # Look for temperature slider
            temp_sliders = self.driver.find_elements(By.XPATH, "//div[@data-testid='stSlider']//input")
            
            for slider in temp_sliders:
                try:
                    # Check if this might be temperature slider
                    parent_text = slider.find_element(By.XPATH, "../../../..").text.lower()
                    if "temp" in parent_text or "°c" in parent_text:
                        # Set slider value using JavaScript
                        self.driver.execute_script(f"arguments[0].value = {temperature}", slider)
                        self.driver.execute_script("arguments[0].dispatchEvent(new Event('input', { bubbles: true }))", slider)
                        print(f"Temperature set to {temperature}°C")
                        time.sleep(2)
                        break
                except:
                    continue
            
            # Look for plot type selectbox
            selectboxes = self.driver.find_elements(By.XPATH, "//div[@data-testid='stSelectbox']//select")
            
            for selectbox in selectboxes:
                try:
                    options = selectbox.find_elements(By.TAG_NAME, "option")
                    for option in options:
                        option_text = option.text.lower()
                        if "activation" in option_text or "ea" in option_text:
                            option.click()
                            print(f"Selected plot type: {option.text}")
                            time.sleep(3)
                            break
                except:
                    continue
            
            print("Settings configured successfully")
            return True
            
        except Exception as e:
            print(f"Settings configuration failed: {e}")
            return False
    
    def extract_streamlit_plotly_data(self):
        """Extract data from Plotly graphs in Streamlit"""
        print("Extracting Plotly data from Streamlit app...")
        
        try:
            # Wait for plots to render
            self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".plotly-graph-div"))
            )
            time.sleep(5)
            
            # Enhanced JavaScript to extract plotly data
            plotly_data = self.driver.execute_script("""
                var plots = document.querySelectorAll('.plotly-graph-div, .js-plotly-plot');
                var extractedData = [];
                
                console.log('Found ' + plots.length + ' plot elements');
                
                for (var i = 0; i < plots.length; i++) {
                    try {
                        var plot = plots[i];
                        console.log('Processing plot ' + i, plot);
                        
                        if (plot._fullData && plot._fullData.length > 0) {
                            var plotInfo = {
                                plotIndex: i,
                                plotTitle: plot._fullLayout && plot._fullLayout.title ? 
                                          (plot._fullLayout.title.text || plot._fullLayout.title) : 'Unknown',
                                xAxisTitle: plot._fullLayout && plot._fullLayout.xaxis ? 
                                           (plot._fullLayout.xaxis.title || {}).text || 'X-Axis' : 'X-Axis',
                                yAxisTitle: plot._fullLayout && plot._fullLayout.yaxis ? 
                                           (plot._fullLayout.yaxis.title || {}).text || 'Y-Axis' : 'Y-Axis',
                                traces: []
                            };
                            
                            plot._fullData.forEach(function(trace, traceIndex) {
                                if (trace.x && trace.y) {
                                    var traceData = {
                                        name: trace.name || 'trace_' + traceIndex,
                                        x: Array.from(trace.x || []),
                                        y: Array.from(trace.y || []),
                                        text: Array.from(trace.text || []),
                                        hovertext: Array.from(trace.hovertext || []),
                                        customdata: trace.customdata ? Array.from(trace.customdata) : [],
                                        type: trace.type || 'scatter',
                                        mode: trace.mode || 'markers',
                                        dataLength: (trace.x || []).length
                                    };
                                    plotInfo.traces.push(traceData);
                                }
                            });
                            
                            if (plotInfo.traces.length > 0) {
                                extractedData.push(plotInfo);
                            }
                        }
                    } catch (e) {
                        console.log('Error extracting plot ' + i + ':', e);
                    }
                }
                
                console.log('Extracted data from ' + extractedData.length + ' plots');
                return extractedData;
            """)
            
            print(f"Successfully extracted data from {len(plotly_data)} plots")
            
            # Log data summary
            for i, plot in enumerate(plotly_data):
                print(f"Plot {i}: {len(plot.get('traces', []))} traces")
                for j, trace in enumerate(plot.get('traces', [])):
                    print(f"  Trace {j}: {trace.get('dataLength', 0)} data points")
            
            return plotly_data
            
        except Exception as e:
            print(f"Data extraction failed: {e}")
            self.driver.save_screenshot("extraction_error.png")
            return []
    
    def close(self):
        """Close the driver"""
        if self.driver:
            self.driver.quit()

# Updated main workflow
def updated_main_workflow(username, password, temp_range=(20, 30), headless=False):
    """Updated main workflow with Streamlit-specific handling"""
    
    scraper = UpdatedDDSEScraper(headless=headless)
    
    try:
        # Start driver
        scraper.start_driver()
        
        # Login with updated method
        if not scraper.streamlit_login(username, password):
            print("Login failed. Please check credentials and page structure.")
            return None
        
        # Configure settings
        target_temp = (temp_range[0] + temp_range[1]) / 2
        if not scraper.configure_streamlit_settings(target_temp):
            print("Failed to configure settings.")
            return None
        
        # Extract data
        raw_data = scraper.extract_streamlit_plotly_data()
        if not raw_data:
            print("No data extracted.")
            return None
        
        print(f"Extraction completed! Found {len(raw_data)} plots with data.")
        
        # Save raw data
        import json
        with open('ddse_streamlit_data.json', 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        return raw_data
    
    except Exception as e:
        print(f"Workflow error: {e}")
        return None
    
    finally:
        scraper.close()

# Test script
if __name__ == "__main__":
    # Replace with your actual credentials
    USERNAME = "your_email@domain.com"
    PASSWORD = "Edunovais#1"
    
    print("Starting updated DDSE 2.0 extraction...")
    data = updated_main_workflow(USERNAME, PASSWORD, headless=False)
    
    if data:
        print("SUCCESS: Data extracted!")
        print(f"Total plots: {len(data)}")
        for i, plot in enumerate(data):
            print(f"Plot {i}: {len(plot.get('traces', []))} traces")
    else:
        print("FAILED: No data extracted")
