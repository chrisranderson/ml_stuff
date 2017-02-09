# This is weird, but I don't know how else to do it.

cp setup.py ../finch_setup_temp.py
cd .. && sudo python finch_setup_temp.py develop
rm finch_setup_temp.py
