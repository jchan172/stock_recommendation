#!/bin/bash

# Remove the existing package directory and create a new one
rm -rf package/
mkdir package

# Install the required packages into the package directory
pip3 install --target ./package requests workalendar statistics datetime pytz

# Change to the package directory and create a zip file
cd package
zip -r ../stock_recommendation_package.zip .

# Go back to the previous directory
cd ..

# Add additional files to the zip package
zip -r stock_recommendation_package.zip slack_messenger.py stock_recommendation

# Clean up - Remove the package directory that was created
rm -rf package/
