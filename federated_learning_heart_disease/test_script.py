from selenium import webdriver
from selenium.webdriver.support.ui import Select
import time 




def classifiers():
    driver = webdriver.Chrome('chromedriver.exe')
    driver.get('http://127.0.0.1:5000/')
    time.sleep(3)

    home_btn = driver.find_element_by_id('btn2')
    home_btn.click()

    age = driver.find_element_by_id('age')
    age.send_keys('54')

    select = Select(driver.find_element_by_id('sex'))
    select.select_by_index(1)

    select = Select(driver.find_element_by_id('cp'))
    select.select_by_index(1)

    bps = driver.find_element_by_id('trestbps')
    bps.send_keys('89')

    chol = driver.find_element_by_id('chol')
    chol.send_keys('69')

    select = Select(driver.find_element_by_id('fbs'))
    select.select_by_index(1)

    select = Select(driver.find_element_by_id('restecg'))
    select.select_by_index(1)

    thalach = driver.find_element_by_id('thalach')
    thalach.send_keys('88')

    exang = driver.find_element_by_id('exang')
    exang.send_keys('55')

    select = Select(driver.find_element_by_id('oldpeak'))
    select.select_by_index(1)

    select = Select(driver.find_element_by_id('slope'))
    select.select_by_index(1)

    select = Select(driver.find_element_by_id('ca'))
    select.select_by_index(1)

    select = Select(driver.find_element_by_id('thal'))
    select.select_by_index(1)

    time.sleep(3)
    submit_button = driver.find_element_by_id('subbtn')
    submit_button.click()
    time.sleep(5)



classifiers()


