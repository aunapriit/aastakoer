# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 2019
@author: Priit

Retrieve data about results of Bernese Mountain dog dogshows held in Estonia and write it to the csv file.
Alter the year and breed info for other use cases.
"""
import csv, time
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

# change year value to get the results from a different year
year = '2018'
breed = 'BERNI ALPI KARJAKOER' # breed name to search from results
breed_lowercase = ''.join(breed.lower())
filename = 'dogshows_' + breed_lowercase + '_' + year + '_lisapunktid' + '.csv'
driver = webdriver.Firefox()
driver.get("https://online.kennelliit.ee/show/")


#delay function
def ootaja():
    #driver.implicitly_wait(3)
    time.sleep(3)
    return

# check if xpath does work and return value or exception
def search_table(filename, dogshow, klass):
    
    ootaja()
    # wait for table loading
    dogname = 'no data'
    result = 'no data'
    
    # select all rows from the results table
    try:
        tablerows = driver.find_elements_by_xpath('//*[@id="results"]/table//tr')
        if len(tablerows) < 2:
            print('1 row table')
            return
    except NoSuchElementException:
        print('no table rows found')
        return # no table found - skip the following
        
    # iteration over table rows and get data from it
    for i in range(2, len(tablerows)+1):
        
        xpath_str = '//*[@id="results"]/table//tr[' + str(i) + ']/td'
        
        try:
            osaleja_jutt = driver.find_element_by_xpath(xpath_str +'[2]').text
            
            if breed in osaleja_jutt:
                
                try:
                    dogname = driver.find_element_by_xpath(xpath_str +'[2]/span').text # search for breed from text
                    print ('dog: ', dogname)
                except NoSuchElementException:
                    print ('no dog name found')
                    continue
                
                try:
                    result = driver.find_element_by_xpath(xpath_str +'[3]/span').text
                    print('place:', result)
                except NoSuchElementException:
                    print ('no result about place')
                    pass
                
                #write found data to csv file
                write_datarow(filename, dogshow, klass, dogname, result)
                
        except NoSuchElementException:
            print ('no information found in table')
            continue
        
    return

def write_datarow(filename, dogshow, klass, dogname, result):
    # makerow of data and write it to the csv file
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({'naitus': dogshow, 'klass': klass, 'koer': dogname, 'koht': result})
    return

# write csv file header and define fieldnames
fieldnames = ['naitus', 'klass', 'koer', 'koht']
with open(filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

number = 0 #counter for testing

# uncomment to select table rows that contains links to all shows
#links = [link.get_attribute("href") for link in driver.find_elements_by_link_text('Tulemused')]

# select table rows that contains links to shows in certain year
linkscollection = driver.find_elements_by_xpath('//tr[contains(string(), "' + year +'")]//a[contains(string(), "Tulemused")]')
links = [link.get_attribute("href") for link in linkscollection]

# iteration over collected links to shows
for link in links:
    ootaja()
    
    driver.get(link) # get page of dogshow

    # heading of the dogshow
    ootaja()
    dogshow = driver.find_element_by_xpath('//*[@id="showname"]').text
    print(dogshow)

    # get group 2 class winners - NB! class nr depends on breed
    try:
        ootaja()
        ruhm_2 = driver.find_element_by_link_text('Rühm 2')
        ruhm_2.click()
        klass = 'Rühm 2'
        search_table(filename, dogshow, klass)
    except NoSuchElementException:
        print ('Rühm 2 link not found')
        #pass # skip and continue
        continue # our breed is missing - go to next link

    # get baby class winners
    try:
        ootaja()
        parim_beebi = driver.find_element_by_link_text('Parim beebi')
        parim_beebi.click()
        klass = 'Parim beebi'
        search_table(filename, dogshow, klass)
    except NoSuchElementException:
        print ('Parim beebi link not found')
        pass # skip and continue
        
    # get puppy class winners
    try:
        ootaja()
        parim_kutsikas = driver.find_element_by_link_text('Parim kutsikas')
        parim_kutsikas.click()
        klass = 'Parim kutsikas'
        search_table(filename, dogshow, klass)
    except NoSuchElementException:
        print ('Parim kutsikas link not found')
        pass # skip and continue
        
    # get junior class winners
    try:
        ootaja()
        parim_juunior = driver.find_element_by_link_text('Parim juunior')
        parim_juunior.click()
        klass = 'Parim juunior'
        search_table(filename, dogshow, klass)
    except NoSuchElementException:
        print ('Parim juunior link not found')
        pass # skip and continue
                
    # get veteran class winners
    try:
        ootaja()
        parim_veteran = driver.find_element_by_link_text('Parim veteran')
        parim_veteran.click()
        klass = 'Parim veteran'
        search_table(filename, dogshow, klass)
    except NoSuchElementException:
        print ('Parim veteran link not found')
        pass # skip and continue

    # get BIS(best in show) class winners
    try:
        ootaja()
        bis = driver.find_element_by_link_text('Näituse parim / BIS')
        bis.click()
        klass = 'BIS'
        search_table(filename, dogshow, klass)
    except NoSuchElementException:
        print ('BIS link not found')
        pass # skip and continue

            
    # use break and counter for testing 
    """if number == 5:
        break
    number = number + 1"""
    
driver.close()
