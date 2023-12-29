#!/bin/bash

. ./run-env.sh

cd ../python
zip -r $APP_NAME-python.zip *
mv $APP_NAME-python.zip ../resources
cd ../resources