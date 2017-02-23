# This is weird, but I don't know how else to do it.

cp setup.py ../finch_setup_temp.py # Use this filename to avoid conflicts.
cd .. && sudo python3 finch_setup_temp.py develop # No way to specify different setup.py for pip.
rm finch_setup_temp.py
