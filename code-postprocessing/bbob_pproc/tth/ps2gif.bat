@echo off
:: ps2gif batch file. Use at your own risk.
:: Requires ghostscript and the netpbm utilities.
if %2.==. goto usage
echo "Calling ghostscript to convert %1 to %2 , please wait ..."
gs -sDEVICE=ppmraw -sOutputFile=- -sNOPAUSE -q %1 -c showpage -c quit | pnmcrop| ppmtogif >%2
if %3.==. goto end
echo "Calling ghostscript to convert %1 to %3 , please wait ..."
gs -sDEVICE=ppmraw -sOutputFile=- -sNOPAUSE -r12 -q %1 -c showpage -c quit | pnmcrop|  ppmtogif >%3
goto end
:usage
echo " Usage: ps2gif <file.ps> <file.gif> [<icon.gif>]"
:end

