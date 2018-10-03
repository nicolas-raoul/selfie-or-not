#
#
#
#
# OUT OF DATE !!!
# Please use run-preprocessed.sh instead
#
#
#
#
python -u run.py 2>&1 |grep -v "Invalid SOS parameters for sequential JPEG" | grep -v "PNG warning"
