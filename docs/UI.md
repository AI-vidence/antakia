* splashScreenLayout
    * antakiaLogo, Image,
    * explainComputationRow, Row
        * explainComputationProgLinear, ProgressLinear
    * projComputationRow
        * projComputationProgLinear, ProgressLinear
* ourGUIVBox
    * menuAppBar
        * logoLayout
            * logoImage
        * HTML = "<h2>Antakia"
        * Spacer
        * backupBtn
        * settingsBtn    
        * figureSizeDialog
            * v.Card
                * CartTitle
                * CardText
                    * figureSizeSliderRow
                        * figureSizeSlider
                        * figureSizeSliderIntText
        * goToWebsiteBtn
    * backupsDialog
    * topButtonsHBox
        * dimSwitchRow
            * Icon, Icon
            * dimSwitch
            * Icon, Icon
        * Layout
            * colorChoiceBtnToggle
            * resetOpacityBtn
            * explanationSelect
            * computeMenu
                * computeExplanationsCard
                    * Tab SHAP
                        * ourSHAPProgLinearColumn
                    * Tab LIME
                        * ourLIMEProgLinearColumn
        * Layout / Layout
            * dimReducVSHBox
                * dimReducVSSelect
                * Layout
                    * dimReducVSSettingsMenu
                        * paCMAPVSParamsBox
                        * allPaCMAPVSSlidersVBox
                            * neighboursPaCMAPVSSliderLayout
                            * mnRatioPaCMAPVSSliderLayout
                            * fpRatioPacMAPVSSliderLayout
                        * allPaCMAPActionsVSHBox
                            * validatePaCMAPParamsVSBtn
                            * resetPaCMAPParamsVSBtn
                * busyVSHBox
                    * ourProgCircular
            * dimReducESHBox
                * dimReducESSelect
                * Layout
                    * dimReducESSettingsMenu
                        * paramsProjESVBox
                            * allPaCMAPESSlidersVBox
                                * neighboursPaCMAPESSliderLayout
                                * mnRatioPaCMAPESSliderLayout
                                * fpRatioPacMAPESSliderLayout
                            * allPaCMAPActionsESHBox
                                * validatePaCMAPParamsESBtn
                                * resetPaCMAPParamsESBtn
                * busyESHBox
                    * ourProgCircular
    * figuresVBox
        * figuresHBox
            * _leftVSFigureVBox
                * _leftVSFigure
            * _rightESFigureVBox
                * _rightESFigure
        ou :
        * figuresHBox
            * _leftVSFigure3DVBox
                * _leftVSFigure3D
            * _rightESFigure3DVBox
                * _rightESFigure3D
    * antakiaMethodCard
        * Tabs
            * Tab
            * Tab
            * Tab
            * Tab
        * CardText
            * WindowItem
                * tabOneSelectionColumn
                    * selectionCard
                    * out_accordion
                    * clusterGrp
                    * loadingClustersProgLinear
                    * clusterResults
            * WindowItem
                * tabTwoSkopeRulesColumn
            * WindowItem
                * tabThreeSubstitutionVBox, VBox
                    * loadingModelsProgLinear
                    * subModelslides, List
                        * slideItem
                            * Card
                                * Row
                                    * Icon
                                    * CardTitle
                                * CardText

            * WindowItem
                * tabFourRegionListVBox
    * magicGUI


