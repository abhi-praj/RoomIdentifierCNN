from selenium import webdriver
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome()
driver.get("https://lsm.utoronto.ca/webapp/f?p=210:1::::::")

wait = WebDriverWait(driver, 20)
wait.until(EC.presence_of_element_located((By.ID, "P1_BLDG")))
driver.implicitly_wait(10)

output = []

building_select = Select(driver.find_element(By.ID, "P1_BLDG"))
building_options = [
    (opt.text.strip(), opt.get_attribute("value"))
    for opt in building_select.options
    if opt.get_attribute("value") and "select" not in opt.text.lower()
]

for building_label, building_value in building_options:
    code_parts = building_label.split(" ", 1)
    if len(code_parts) < 2:
        continue
    building_code = code_parts[0]

    building_select = Select(driver.find_element(By.ID, "P1_BLDG"))
    building_select.select_by_value(building_value)

    time.sleep(1.5)

    wait.until(EC.presence_of_element_located((By.ID, "P1_ROOM")))
    room_select = Select(driver.find_element(By.ID, "P1_ROOM"))

    for room_opt in room_select.options:
        room = room_opt.text.strip()
        if not room or "select" in room.lower():
            continue
        output.append(f"{building_code}{room}")

driver.quit()

# Output the list
for entry in output:
    print(entry)