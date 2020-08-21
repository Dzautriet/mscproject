for %%i in (0.1 0.5 0.9) do (
	for %%j in (0.1 0.5 0.9) do (
		python reg_main_3.py %%i %%j mnist
	)
)