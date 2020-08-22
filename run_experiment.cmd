for %%i in (0.3 0.4 0.4) do (
	for %%j in (0.3 0.4 0.5) do (
		python reg_main_3.py %%i %%j mnist
	)
)