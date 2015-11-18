@echo off
::l2h.bat latex to HTML script to call TtH.
if %1.==. goto usage
tth <%1.tex >%1.html -L%1 %2 %3 %4 %5 %6
goto end
:usage
echo  Usage: l2h file [switches]
:end
