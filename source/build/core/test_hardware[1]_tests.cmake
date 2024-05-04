add_test([=[HardwareTest.getNumProcessorsOnln]=]  /data/home/liuhanyu/hyliu/code/AI2Pot/source/build/bin/core/test_hardware [==[--gtest_filter=HardwareTest.getNumProcessorsOnln]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[HardwareTest.getNumProcessorsOnln]=]  PROPERTIES WORKING_DIRECTORY /data/home/liuhanyu/hyliu/code/AI2Pot/source/build/core SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_hardware_TESTS HardwareTest.getNumProcessorsOnln)
