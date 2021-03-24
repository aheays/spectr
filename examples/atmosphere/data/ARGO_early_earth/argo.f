
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c	ARGO.F (borne of NAHOON.F)
c
c	There is no file with the differential equations explicitly
c	written but some loops. For that an additional species which is 
c	blank is added.
c
c	VODE integretor
c	Wakelam V. Mai. 2006
c       
c       Modified by P. B. Rimmer to include reverse reactions, photochemistry
c       for arbitrary UV fields, condensation of major species, and
c       simple estimates for molecular and eddy diffusion.
c
c       Rimmer & Helling 2015
c
c       Modification for the High Temperature by N. Harada, Sep. 2010
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


        PROGRAM NAHOON_HT

        IMPLICIT NONE

        INTEGER NS,NRES,NRTOT,NROUT,NTIME,ITYPE,INUM,IPROD,NELEM,NS2,
     &          ITEST,ITEST2,ELEMENT,I,J,ISP,JSTEP,L,NPLOT,JJ,IH,
     &          RADNUM,WLNUM,PRNUM,THRESHOLD
c	NS = number of species in cond_init + number of species in reservoir +1
        PARAMETER (NS=928+5+1,NRES=5)
c       NRTOT = number of reactions from the network file + number of outgas.dat reactions
        PARAMETER (NRTOT=6612+5,NROUT=5)
        PARAMETER (NTIME=299,NELEM=15+1,NPLOT=NS-1)
c	THRESHOLD = number of species before banking
        PARAMETER (WLNUM=10001,PRNUM=209+1,THRESHOLD=510)
        DOUBLE PRECISION SN,Y,AB,PLOTAB,TPLOT,TD,TDABOVE,TDBELOW,TEL,
     &          TIMERES,NHTOT,NHABOVE,NHBELOW,TYR,XK1,XK2,YGRAIN,XCO,
     &          XH2,A,B,C,RANDOM1,UNC,ZETA,ABHOLD,GTODN,SNGRAIN,
     &          STCOEFF,SIGMA,FIELD,TSCALE,VZ,VZABOVE,VZBELOW,ZNOW,
     &          ZPREV,ZNEXT,RD,DTOGM,RHOD,MUAV,SIGMAAV,LOGG,ABTOTAL,
     &          QERROR,ABPLUS,ABMINUS,KB,MP,PI,MOLE,SHEIGHT,LIFE
        CHARACTER*8 SPEC,REACTANT*8
        DIMENSION SN(NS),Y(NS),AB(NS),PLOTAB(NTIME,NPLOT),A(NRTOT),
     &            B(NRTOT),C(NRTOT),TIMERES(NTIME),SPEC(NS),ITEST(14),
     &            ELEMENT(NELEM,NS),REACTANT(3,NRTOT),ITYPE(NRTOT),
     &            RANDOM1(NRTOT),UNC(NRTOT),RADNUM(NRTOT),
     &            SIGMA(PRNUM,WLNUM),FIELD(2,WLNUM),
     &            LIFE(NRES)
        CHARACTER AA*78,RANDMODE,PHOTOMODE,DIFFMODE,CONDENSE,COLDTRAP,
     &            ISVENUS,ISISOPRENE,DIRECTION*2,ROTATION,CONDSPEC
        DATA MP,KB,PI,MOLE
     &     /1.67262D-24, 1.38064D-16, 3.14159265, 6.023D23/

        INTEGER REACT(NRTOT,7),NRBIS
        DOUBLE PRECISION K(NRTOT)
        COMMON/EQUA/K,REACT,NRBIS


ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c	Variable declaration for the REVERSE REACTIONS
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        INTEGER TCE_SWITCH,NPOLY
        DOUBLE PRECISION TC,ADJUST_TIMESTEP,ADJ1,ADJ2
        PARAMETER (NPOLY=14,ADJ2=0.0D+00)
        DIMENSION TC(NPOLY,NS)


ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 	Variable declaration FOR THE DLSODE SUBROUTINE
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        EXTERNAL F, JAC
        INTEGER NEQ, ITOL, ITASK, ISTATE, IOPT, LRW, IWORK, LIW, JEX, MF
        DOUBLE PRECISION T, TOUT, RTOL, ATOL, RWORK
        PARAMETER (NEQ=NS, LIW=20+NEQ, LRW=22 +  9*NEQ + NEQ**2)
        DIMENSION IWORK(LIW), RWORK(LRW)

        DATA ITOL, ITASK, ISTATE, IOPT, MF, RTOL, ATOL
     &       /1, 3, 1, 1, 22, 1.D-4, 1.D-40/

        NRBIS=NRTOT
        NS2=NS
        IWORK(6)=10000
        RWORK(6)=3.154E+15
        IWORK(5)=5

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c	Files declaration
c	Here, we open the files we have to read (list of species + inital 
c	abundances + reactions + CO and H2 self-shielding factors + random numbers for 
c	the uncertainty calculation) and we define the output files:
c	- Kout.dat contain the values of the rate coefficients
c	- plot.dat the abundances as a function of time
c	- verif.dat contains some balance verifications and the 
c	rates of formation and destruction for each species at 
c	specific times
c	- output.dat is the chemical composition at one time (by default 1e8yr)
c	in the same format as cond_initial.dat
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        OPEN (UNIT=1,FILE='Stand-2020October',STATUS='OLD')
        OPEN (UNIT=2,FILE='input.dat',STATUS='OLD')
        OPEN (UNIT=3,FILE='input_parameter.dat',STATUS='OLD')
        OPEN (UNIT=4,FILE='timeres.dat',STATUS='OLD')
        OPEN (UNIT=5,FILE='random.dat',STATUS='OLD')

        OPEN (UNIT=9,FILE='Kout.dat',STATUS='UNKNOWN')
        OPEN (UNIT=10,FILE='plot.dat',STATUS='UNKNOWN')
        OPEN (UNIT=11,FILE='verif.dat',STATUS='UNKNOWN') 
        OPEN (UNIT=12,FILE='output.dat',STATUS='UNKNOWN') 

        OPEN (UNIT=13,FILE='photo-sigma.dat',STATUS='OLD')
        OPEN (UNIT=14,FILE='blocked.dat',STATUS='OLD')

        OPEN (UNIT=15,FILE='reservoir.dat',STATUS='OLD')
        OPEN (UNIT=16,FILE='outgas.dat',STATUS='OLD')
        OPEN (UNIT=17,FILE='Tout.dat',STATUS='UNKNOWN')

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC 
C	read data files 
c
C	AB is the abundance species with respect to total 
C	SN is in number density (cm-3) 

c	First put some numbers to zero
        DO I=1,NS
                SN(I)=0.D0
                Y(I)=0.D0
                AB(I)=0.D0
                DO ISP=1,NELEM
                   ELEMENT(ISP,I)=0
                ENDDO
        ENDDO

        DO I=1,NRTOT
            K(I) = 0.0D0
        ENDDO

        T=0.D0
        JSTEP=0

C	read output times
        READ(4,*) TIMERES

C	reacd initial conditions (abundance /H) 

        DO I=1,10
           READ (2,*) AA
        ENDDO
        DO I=1,NS-NRES-1
                READ (2,1) SPEC(I),(ELEMENT(ISP,I),ISP=1,NELEM),AB(I),
     &                     (TC(ISP,I),ISP=1,NPOLY)
        ENDDO
        DO I=1,10
           READ (15,*) AA
        ENDDO
        DO I=1,NRES
               READ (15,12) SPEC(I+NS-NRES-1),(ELEMENT(ISP,I+NS-NRES-1),
     &                           ISP=1,NELEM),AB(I+NS-NRES-1),LIFE(I)
        ENDDO
1       FORMAT(5X,A8,1X,16(I3),2X,D10.4,14(1X,D11.4))
12      FORMAT(5X,A8,1X,16(I3),2X,D10.4,2X,D10.4)


C	for the additional blank species 		 	
        SPEC(NS)='        '
        AB(NS)=0.D0
        SN(NS)=0.D0

C	read the parameters 
        DO I=1,9
                READ (3,14) AA
        ENDDO
14      FORMAT(A78)
15      FORMAT(A2)
        READ(3,2) RANDMODE
        READ(3,2) DIFFMODE
        READ(3,2) PHOTOMODE
        READ(3,2) CONDENSE
        READ(3,2) COLDTRAP
        READ(3,2) ISVENUS
        READ(3,2) ISISOPRENE
        READ(3,15) DIRECTION
        READ(3,2) ROTATION
2       FORMAT(A1)
        READ(3,3) NHTOT
        READ(3,3) NHABOVE
        READ(3,3) NHBELOW
        READ(3,3) TD
        READ(3,3) TDABOVE
        READ(3,3) TDBELOW
        READ(3,3) TEL
        READ(3,3) VZ
        READ(3,3) VZABOVE
        READ(3,3) VZBELOW
        READ(3,3) ZNOW
        READ(3,3) ZPREV
        READ(3,3) ZNEXT
3       FORMAT(D9.3)

C	if we are in the uncertainty mode, RANDMODE is Y 
c	and the temperature and density are read in the random.dat file
        IF (RANDMODE.EQ.'Y') THEN
                READ(5,*) NHTOT
                READ(5,*) TD
        ENDIF

C	compute the species density from the initial abundances 
c	SNGRAIN is the total abundance of grains
        DO I=1,NS-1
                SN(I)=AB(I)*NHTOT
                PLOTAB(1,I)=SN(I) 
        ENDDO


C	read the chemical database
        CALL READ_REACT(NS2,RANDMODE,UNC,ELEMENT,NELEM,A,B,C,REACTANT,
     &                  SPEC,ITYPE,RANDOM1,ZETA,GTODN,RADNUM,RD,DTOGM,
     &                  RHOD,MUAV,SIGMAAV,LOGG)

c        WRITE(*,*) 'MUAV BEFORE =', MUAV
        CALL SETDIFF(ELEMENT,SN,MUAV,DIRECTION)
c        WRITE(*,*) 'MUAV AFTER =', MUAV
        
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c       read the cross-section file for the photochemistry
        IF (PHOTOMODE.EQ.'Y') CALL PHOTOREAD(SIGMA,FIELD)

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c	compute the rate coefficients
c

        CALL RATE_COEFF(A,B,C,GTODN,ZETA,NHTOT,NHABOVE,NHBELOW,NS2,SN,
     &                  SPEC,TD,TDABOVE,TDBELOW,TEL,RANDOM1,JSTEP,ITYPE,
     &                  RANDMODE,UNC,ELEMENT,REACTANT,SNGRAIN,STCOEFF,
     &                  TC,RADNUM,SIGMA,FIELD,PHOTOMODE,DIFFMODE,
     &                  DIRECTION,ROTATION,ZNOW,ZPREV,ZNEXT,VZ,VZABOVE,
     &                  VZBELOW,RD,DTOGM,RHOD,MUAV,SIGMAAV,LOGG,
     &                  CONDENSE,ISVENUS,ISISOPRENE,LIFE)

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c	compte the chemical evolution
c
     
C	definition of the times at which the rates of formation/destruction
c	are writen in verif.dat
        ITEST(1) = 2
        ITEST(2) = 11 
        ITEST(3) = 34
        ITEST(4) = 56
        ITEST(5) = 79
        ITEST(6) = 101
        ITEST(7) = 121
        ITEST(8) = 141
        ITEST(9) = 161
        ITEST(10) = 181
        ITEST(11) = 201
        ITEST(12) = 221
        ITEST(13) = 241
        ITEST(14) = 281
        ITEST2=1

C	because of the definition of the rate coefficient of H2 formation 
c	we have to devide the rate coefficient of this reaction by N(H). 
c 	Here, we first save the value of the rate
c	XK1=K(IH2FORM)

C	starting the loop 

        WRITE (*,*) 'It is computing.... wait !'

        ADJ1=1.0D+7
        IF (NHTOT.GE.1.0D+15) ADJ1=1.0D+7
        IF (NHTOT.GE.1.0D+16) ADJ1=1.0D+6
        IF (NHTOT.GE.1.0D+17) ADJ1=1.0D+5
        IF (NHTOT.GE.1.0D+18) ADJ1=1.0D+4
        IF (NHTOT.GE.1.0D+19) ADJ1=1.0D+3
        IF (NHTOT.GE.1.0D+20) ADJ1=1.0D+2
        ADJUST_TIMESTEP=ADJ1/NHTOT*EXP(-TD/5.0D+2)
c        ADJUST_TIMESTEP=ADJUST_TIMESTEP*1.D+8
c	IF (PHOTOMODE.EQ."Y") ADJUST_TIMESTEP=ADJUST_TIMESTEP*1D+05
c     &					      *(NHTOT/1.0D+19)**0.714

        IF (VZ.GT.0.D+0) TSCALE = 0.5D-1*ZNEXT/VZ
        IF (VZ.LE.0.D+0) TSCALE = 1.0D+99

c       IN THE CASE THAT CERTAIN (MAYBE BAD) ASSUMPTIONS ARE BEING MADE
c       ABOUT THE EDDY DIFFUSION COEFFICIENT, USE THIS APPROX INSTEAD
c       TO COMPARE RESULTS:
c        IF (VZ.GT.0.D+0) THEN 
c          SHEIGHT = KB*TD/(MUAV*MP*1.D+1**LOGG)
c          IF (ZNEXT/SHEIGHT.LE.5.D-1) TSCALE = TSCALE*ZNEXT/SHEIGHT
c        ENDIF

c       LET DYNAMICS SET THE MINIMUM TIMESCALE:
        IF ((VZ.GT.0.D+0).AND.(ADJUST_TIMESTEP.GT.
     &          TSCALE/TIMERES(NTIME-1))) THEN
                ADJUST_TIMESTEP=TSCALE/TIMERES(NTIME-1)
        ENDIF

c       TURN THIS OFF UNLESS YOU WANT TO DO DEEP DYNAMICS! 
c        IF (VZ.GT.0.D+0) THEN
c                ADJUST_TIMESTEP=TSCALE/TIMERES(NTIME-1)
c        ENDIF

        DO JSTEP=2,NTIME

C	TOUT is the output time of the dlsode subroutine and is set to be the 
c	time in sec from the table timeres 
c	TYR is the time in year writen in the output file
        TOUT=TIMERES(JSTEP)*3.154D7*ADJUST_TIMESTEP
        TYR=TIMERES(JSTEP)*ADJUST_TIMESTEP

C	write the verif.dat file, the output.dat file 
        IF (JSTEP.EQ.2) THEN 
                WRITE(*,*) 'T(SEC)= 0.0'
                WRITE(11,*) 'T(SEC)= 0.0'

                CALL CHECKING(NS2,ELEMENT,NELEM,SN)
                CALL COMP_RATES(SN,SPEC,NEQ,0)
        ENDIF

C	we transfer the values of the species densities into the 
c	variable Y used by dlsode	
        DO ISP=1,NS
                Y(ISP) = SN(ISP)
        ENDDO

C	no variation for the blank species
        Y(NS)=1.D0

C	we call the subroutine to solve the differential equations 
        CALL DLSODE (F, NEQ, Y, T, TOUT, ITOL, RTOL, ATOL, ITASK,
     &                ISTATE, IOPT, RWORK, LRW, IWORK, LIW, JAC, MF)

C	put the densities as abundances compared to total H 
        DO ISP=1,NS
                SN(ISP) = Y(ISP)
                IF (SN(ISP).LT.0.D+0) SN(ISP)=0.D+0
        ENDDO

        CALL RATE_COEFF(A,B,C,GTODN,ZETA,NHTOT,NHABOVE,NHBELOW,NS2,SN,
     &                  SPEC,TD,TDABOVE,TDBELOW,TEL,RANDOM1,JSTEP,ITYPE,
     &                  RANDMODE,UNC,ELEMENT,REACTANT,SNGRAIN,STCOEFF,
     &                  TC,RADNUM,SIGMA,FIELD,PHOTOMODE,DIFFMODE,
     &                  DIRECTION,ROTATION,ZNOW,ZPREV,ZNEXT,VZ,VZABOVE,
     &                  VZBELOW,RD,DTOGM,MUAV,RHOD,SIGMAAV,LOGG,
     &                  CONDENSE,ISVENUS,ISISOPRENE,LIFE)
        
        DO ISP=1,NS
                AB(ISP) = SN(ISP)/NHTOT
                IF (SPEC(ISP).EQ.'CO      ') XCO=AB(ISP)
                IF (SPEC(ISP).EQ.'H2      ') XH2=AB(ISP)
        ENDDO
 
C	writing the abundances in the PLOTAB for the plot.dat file
        DO ISP=1,NS-1
            PLOTAB(JSTEP,ISP)=SN(ISP)
        ENDDO

C	write the verif.dat file, the output.dat file 
        IF (JSTEP.EQ.ITEST(ITEST2)) THEN 
                WRITE(*,*) 'T(SEC)=',TOUT
                WRITE(11,*) 'T(SEC)=',TOUT

                CALL CHECKING(NS2,ELEMENT,NELEM,SN)
                CALL COMP_RATES(SN,SPEC,NEQ,0)
                ITEST2=ITEST2+1
        ENDIF

        IF ((TOUT.GT.TSCALE).OR.(JSTEP.EQ.NTIME)) THEN
        
        WRITE(*,*) 'T(SEC)=',TOUT
        WRITE(11,*) 'T(SEC)=',TOUT

        CALL CHECKING(NS2,ELEMENT,NELEM,SN)
        CALL COMP_RATES(SN,SPEC,NEQ,1)

        

        WRITE(12,4) 'cccccccccccccccccccccccccccccccccccccccccccccccccc'
        WRITE(12,4) 'c output.dat'
        WRITE(12,4) 'c'
        WRITE(12,4) 'c	output file of the chemical composition'
        WRITE(12,4) 'c	Mixing ratios'
        WRITE(12,4) 'c	Columns indicate:'
        WRITE(12,4) 'c	number	species	 charge  element Abundance'
        WRITE(12,4) 'c	Please respect the format'
        WRITE(12,4) 'cccccccccccccccccccccccccccccccccccccccccccccccccc'
        WRITE(12,5) TOUT,'        +- H  He C  N  O  Si S  Fe Na Mg Cl P 
     &K  Ar Ti'
4       FORMAT(78A)
5       FORMAT(D8.2,70A)
                ABTOTAL = 0.D+0
                ABPLUS = 0.D+0
                ABMINUS = 0.D+0
                QERROR = 0.D+0
                DO I=1,NS-NRES-1
                  IF (AB(I).LT.1.D-99) AB(I)=0.D+0
                  CONDSPEC=SPEC(I)
                  IF((CONDSPEC.EQ.'J').AND.(COLDTRAP.EQ.'Y')
     &            .AND.DIRECTION.EQ.'UP') THEN
                    AB(I)=0.D+0
                  ENDIF
                ENDDO
                DO I=1,NS-NRES-1
                  IF (ELEMENT(1,I).EQ.-1) THEN
                    QERROR = QERROR - AB(I)
                    ABMINUS = ABMINUS + AB(I)
                  ENDIF                 
                  IF (ELEMENT(1,I).EQ.1) THEN
                    QERROR = QERROR + AB(I)
                    ABPLUS = ABPLUS + AB(I)
                  ENDIF
                ENDDO

                IF ((ABPLUS.GT.1.D-30).OR.(ABMINUS.GT.1.D-30)) THEN
                  IF (QERROR.GE.0.D+0) THEN
                    DO I=1,NS-NRES-1
                      IF (ELEMENT(1,I).EQ.1) 
     &                  AB(I) = AB(I)/(ABMINUS + QERROR)*ABMINUS
                    ENDDO
                  ENDIF

                  IF (QERROR.LT.0.D+0) THEN
                    DO I=1,NS-NRES-1
                      IF (ELEMENT(1,I).EQ.-1) 
     &                  AB(I) = AB(I)/(ABPLUS - QERROR)*ABPLUS
                    ENDDO
                  ENDIF
                ENDIF


                DO I=1,THRESHOLD
                  ABTOTAL = ABTOTAL + AB(I)
                ENDDO
                
                DO I=1,NS-NRES-1
                  WRITE(12,6) I,SPEC(I),(ELEMENT(ISP,I),ISP=1,NELEM),
     &                        AB(I)/ABTOTAL,(TC(ISP,I),ISP=1,NPOLY)
                ENDDO
        IF (TOUT.GT.TSCALE) WRITE(*,*) "QUENCH!"
        GOTO 77
        ENDIF                           
6       FORMAT(I4,1X,A8,1X,16(I3),2X,D10.4,14(D12.4))

        ENDDO
 
C	end of big loop

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C	write the plot.dat file for all times 

77      WRITE (10,7) 'Gas temperature (K)', TD
        WRITE (10,7) 'Total H density (cm-3)',NHTOT 
7       FORMAT(A23,E10.2)
        WRITE(10,8) (TIMERES(I)*ADJUST_TIMESTEP*3.154D7,I=1,NTIME)
8       FORMAT(8X,299(2X,D14.8))
        DO ISP=1,NS-1
                WRITE(10,9) SPEC(ISP),(PLOTAB(I,ISP),I=1,NTIME)
        ENDDO
9       FORMAT(A8,299(2X,D14.8))

        STOP 
        END 
 
C	end of main program
cccccccccccccccccccccccccccccccccccccccccccccccccccccc

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C	this subroutine reads the reactions in the database 

        SUBROUTINE READ_REACT(NS2,RANDMODE,UNC,ELEMENT,NELEM,A,B,C,
     &             REACTANT,SPEC,ITYPE,RANDOM1,ZETA,GTODN,RADNUM,RD,
     &             DTOGM,RHOD,MUAV,SIGMAAV,LOGG)

        IMPLICIT NONE

        DOUBLE PRECISION A,B,C,RANDOM1,ZETA,DTOGM,RD,RHOD,GTODN,SUM1,
     &          SUM2,AMH,BOLTZ,PI,MOLE,KZERO, KINF,UNC,MUAV,SIGMAAV,
     &          LOGG
        INTEGER INUM,ITYPE,NRBIS2,NROUT,ELEMENT,NELEM,IR1,IR2,IR3,
     &          DES1,DES2,IPROD1,IPROD2,IPROD3,IPROD4,L,NS2,I,J,
     &          RADNUM,JJ

        PARAMETER (NRBIS2=6612+5,NROUT=5)
        CHARACTER REACTANT*8,PRODUIT*8,SPEC*8,AA*78,RANDMODE,SPEC_DATA*8
        DIMENSION ITYPE(NRBIS2),A(NRBIS2),B(NRBIS2),C(NRBIS2), 
     &            INUM(NRBIS2),RANDOM1(NRBIS2),UNC(NRBIS2),
     &            ELEMENT(NELEM,NS2),REACTANT(3,NRBIS2),
     &            PRODUIT(4,NRBIS2),SPEC(NS2),
     &            RADNUM(NRBIS2)
     
        INTEGER REACT(NRBIS2,7),NRBIS
        DOUBLE PRECISION K(NRBIS2)
        COMMON/EQUA/K,REACT,NRBIS

        DATA AMH,BOLTZ,PI,MOLE
     &     /1.66043D-24, 1.38054D-16, 3.14159265, 6.023D23/

C	reading the parameters for the uncertainties if we are in the uncertainty mode 
        IF (RANDMODE.EQ.'Y') THEN
                READ(5,*) RANDOM1
                READ(5,*) UNC
        ENDIF

C	reading other parameters 
        READ(3,10) ZETA
        READ(3,10) DTOGM
        READ(3,10) RD
        READ(3,10) RHOD
        READ(3,10) MUAV
        READ(3,10) SIGMAAV
        READ(3,10) LOGG

10      FORMAT (D9.3)

c	reading the chemical database
        DO J=1,NRBIS2-NROUT
                READ (1,11) REACTANT(1,J),REACTANT(2,J),REACTANT(3,J),
     &                      PRODUIT(1,J),PRODUIT(2,J),PRODUIT(3,J),
     &                      PRODUIT(4,J),RADNUM(J),A(J),B(J),C(J),
     &                      ITYPE(J),INUM(J)

        ENDDO
        DO J=1,NROUT
           JJ = J + NRBIS - NROUT
           READ (16,11) REACTANT(1,JJ),REACTANT(2,JJ),REACTANT(3,JJ),
     &                  PRODUIT(1,JJ),PRODUIT(2,JJ),PRODUIT(3,JJ),
     &                  PRODUIT(4,JJ),RADNUM(JJ),A(JJ),B(JJ),C(JJ),
     &                  ITYPE(JJ),INUM(JJ)
        ENDDO

11      FORMAT (7(A8),3X,I3,2X,3(E9.2),I2,15X,I4) 

C       GTODN is the ratio between the density of gas and the density of grains 
        GTODN=(4.d0*PI*RHOD*RD*RD*RD)/(3.d0*DTOGM*AMH)

c	construct the table REACT which is used to construct the differential equations
        DO I=1,NRBIS2
                DO J=1,NS2
                DO L=1,3
                        IF (REACTANT(L,I).EQ.SPEC(J)) REACT(I,L)=J
                ENDDO
                DO L=1,4
                        IF (PRODUIT(L,I).EQ.SPEC(J)) REACT(I,L+3)=J
                ENDDO
                ENDDO
        ENDDO

C	check the balance of the reactions (elements and charges) 
        DO I=1,NRBIS2
                IR1=REACT(I,1)
                IR2=REACT(I,2)
                IR3=REACT(I,3)
                IPROD1=REACT(I,4)
                IPROD2=REACT(I,5)
                IPROD3=REACT(I,6)
                IPROD4=REACT(I,7)
                DO J=1,NELEM
                    SUM1=ELEMENT(J,IR1)+ELEMENT(J,IR2)+ELEMENT(J,IR3)
                    SUM2=ELEMENT(J,IPROD1)+ELEMENT(J,IPROD2)+
     &                   ELEMENT(J,IPROD3)+ELEMENT(J,IPROD4)
                    IF (SUM1.NE.SUM2) 
     &                 WRITE(*,*) 'BALANCE PROBLEM AT THE REACTION',I
                ENDDO
        ENDDO
16      FORMAT(A8,F7.3,7X,F7.3)

        RETURN
        END 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C	 this subroutine computes the rate coefficients of the reactions 
C	
C	the formula to compute the rates depends on the type of reaction 
C	We have 12 kinds of reactions: 
C	
C
c 	ITYPE 0 3-body reactions
C 	ITYPE 1 cosmic rays and photodis induced reactions
C 	ITYPE 2 forward ion-neutral reactions
c	ITYPE 3 reverse ion-neutral reactions
C 	ITYPE 4 negative-neutral association Electronic recombination
C 	ITYPE 5 3-body recombination
C 	ITYPE 6 thermal ionization
C 	ITYPE 7 neutral-neutral reactions
C 	ITYPE 8 neutral-neutral radiative association    
C 	ITYPE 9 electronic dissociative recombination 
c	ITYPE 10 dissociative electronic recombination 
c	ITYPE 11 3-body ion-neutral
c	ITYPE 12 electron attachment
c	ITYPE 13 photodissociations
c	ITYPE 14 2-body neutral-neutral reverse reactions
c	ITYPE 15 3-body neutral-neutral reverse
c	ITYPE 16 3-body ion-neutral reverse
c	ITYPE 17 Evaporation from surface
c	ITYPE 18 Settling onto surface
c       ITYPE 19 2-body reactions with pressure-dependent branching ratios 
c       ITYPE 20 2-body reverse reactions with pressure-dependent branchinc ratios

c       ITYPE 42 Outgassing fluxes, folded into a rate constant
c       ITYPE 43 Deposition velocities, folded into a rate constant

c       ITYPE 88 Termination reactions for large species (ONLY IF THEY BUILD UP, OTHERWISE LEAVE OFF!)
c	ITYPE 90 Escape from atmosphere (NOT YET IMPLEMENTED!)
c       ITYPE 99 Hypothetical glycine reactions (I TYPICALLY LEAVE THESE OFF)

C	if you add a new kind of reaction, you have to add here the way 
c	to compute the rate coefficient 

 
        SUBROUTINE RATE_COEFF(A,B,C,GTODN,ZETA,NHTOT,NHABOVE,NHBELOW,
     &               NS2,SN,SPEC,TD,TDABOVE,TDBELOW,TEL,RANDOM1,JSTEP,
     &               ITYPE,RANDMODE,UNC,ELEMENT,REACTANT,SNGRAIN,
     &               STCOEFF,TC,RADNUM,SIGMA,FIELD,PHOTOMODE,DIFFMODE,
     &               DIRECTION,ROTATION,ZNOW,ZPREV,ZNEXT,VZ,VZABOVE,
     &               VZBELOW,RD,DTOGM,RHOD,MUAV,SIGMAAV,LOGG,CONDENSE,
     &               ISVENUS,ISISOPRENE,LIFE)

        IMPLICIT NONE

        DOUBLE PRECISION MOLE,GTODN,TC,SN,TEL,PI,AMH,BOLTZ,TT,TD,
     &          TDABOVE,TDBELOW,ZETA,NHTOT,NHABOVE,NHBELOW,A,B,C,TAU,
     &          RANDOM1,UNC,SNGRAIN,NCOLLH2,XH2,XCO,TETA,NCOLLCO,TETA1,
     &          TETA2,TETA3,X,STCOEFF,KZERO,KINF,SIGMA,FIELD,ZNOW,
     &          ZPREV,ZNEXT,VZ,VZABOVE,VZBELOW,KOUT,RD,DTOGM,NSITES,
     &          MUAV,SIGMAAV,LOGG,MASSJ,SIGMAJ,KBANK,KGO,RHOD,PGAS,PMIN,
     &          TCOND,NCOND,NJCOND,NTOT,TMIN,TMAX,FMAX,FACTORF,FACTORC,
     &          FACTORN,ALPHAT,NCOLL,TCOLL,TEMPCOLL,NCOUNT,KOUTGAS,LIFE,
     &          LIFENOW
        INTEGER ITYPE,INUM,NRBIS2,NS2,L,JSTEP,I,J,ISP,JJ,JJD,RADNUM,
     &          PRNUM,RADNUMNOW,NPOLY,ELEMENT,NELEM,IR1,IR2,IR3,BARFLAG,
     &          WLNUM,NS,NRES
        PARAMETER (NELEM=15+1,NS=928+5+1,NRES=5)
        PARAMETER (NRBIS2=6612+5,WLNUM=10001,PRNUM=209+1,NPOLY=14)
        CHARACTER REACTANT*8,RANDMODE,AA*78,PHOTOMODE,DIFFMODE,
     &            DIRECTION*2,ROTATION,CONDENSE,CONDSPEC,ISVENUS,
     &            ISISOPRENE
        CHARACTER*8 SPEC
        DIMENSION ITYPE(NRBIS2),A(NRBIS2),B(NRBIS2),C(NRBIS2), SN(NS2),
     &            REACTANT(3,NRBIS2),RANDOM1(NRBIS2),UNC(NRBIS2),
     &            TC(NPOLY,NS2),RADNUM(NRBIS2),SIGMA(PRNUM,WLNUM),
     &            FIELD(2,WLNUM),ELEMENT(NELEM,NS2),LIFE(NRES),SPEC(NS)

        DOUBLE PRECISION ETAPROD,ETAREACT,GFORM_PROD,GFORM_REACT,
     &          A1_PROD,A2_PROD,A3_PROD,A4_PROD,A5_PROD,A6_PROD,A7_PROD,
     &          A1_REACT,A2_REACT,A3_REACT,A4_REACT,A5_REACT,A6_REACT,
     &          A7_REACT,TC_A1,TC_A2,TC_A3,TC_A4,TC_A5,TC_A6,TC_A7,
     &          TC_B1,TC_B2,TC_B3,TC_B4,TC_B5,TC_B6,TC_B7,
     &          TC_C1,TC_C2,TC_C3,TC_C4,TC_C5,TC_C6,TC_C7,
     &          TC_D1,TC_D2,TC_D3,TC_D4,TC_D5,TC_D6,TC_D7,
     &          TC_E1,TC_E2,TC_E3,TC_E4,TC_E5,TC_E6,TC_E7,
     &          TC_F1,TC_F2,TC_F3,TC_F4,TC_F5,TC_F6,TC_F7,
     &          TC_G1,TC_G2,TC_G3,TC_G4,TC_G5,TC_G6,TC_G7,
     &          GREACTION,TDP,RATEMAX,PHOTORATE,NEWEXPFACTOR,
     &          PVAP,NVAP,OLDCRIT,PHI,RESFRAC,KCOND


        INTEGER REACT(NRBIS2,7),NRBIS
        DOUBLE PRECISION K(NRBIS2)
        COMMON/EQUA/K,REACT,NRBIS

        DATA AMH,BOLTZ,PI,MOLE
     &     /1.66043D-24, 1.38054D-16, 3.14159265, 6.023D23/

        TT=TD/3.0D+2

c       sticking coefficient for Silicates
        STCOEFF=1/(1+0.04*TD**0.5+2.0e-3*TD+8.0e-6*TD**2.0)

c	start computing the rate coefficients depending on the type of reaction 
        DO J=1,NRBIS2

c	   COSMIC RAY IONIZATION:
           IF (ITYPE(J).EQ.1) K(J)=A(J)*ZETA
   
c	   This is the thermochemistry reverse rate coefficients
c	   Each species has a value of HF and SF, calculated using
c	   seven coefficients. There can be three reactants and four
c          products. Since these are reverse reactions, we take the
c	   constants of the reactants minus constants of the products.
c          So the reactants get 
c          
c          A + B + C -> D + E + F + G
c          
c          HF(A) and SF(A) given by TC_A1, TC_A2, TC_A3, TC_A4, TC_A5, TC_A6, TC_A7
c          HF(B) and SF(B) given by TC_B1, TC_B2, TC_B3, TC_B4, TC_B5, TC_B6, TC_B7
c          HF(C) and SF(C) given by TC_C1, TC_C2, TC_C3, TC_C4, TC_C5, TC_C6, TC_C7
c          HF(D) and SF(D) given by TC_D1, TC_D2, TC_D3, TC_D4, TC_D5, TC_D6, TC_D7
c          HF(E) and SF(E) given by TC_E1, TC_E2, TC_E3, TC_E4, TC_E5, TC_E6, TC_E7
c          HF(F) and SF(F) given by TC_F1, TC_F2, TC_F3, TC_F4, TC_F5, TC_F6, TC_F7
c          HF(G) and SF(G) given by TC_G1, TC_G2, TC_G3, TC_G4, TC_G5, TC_G6, TC_G7

           TDP = TD

           IF ((ITYPE(J).EQ.0).OR.(ITYPE(J).GT.0)) THEN
                TC_A1=0.0
                TC_A2=0.0
                TC_A3=0.0
                TC_A4=0.0
                TC_A5=0.0
                TC_A6=0.0
                TC_A7=0.0
                TC_B1=0.0
                TC_B2=0.0
                TC_B3=0.0
                TC_B4=0.0
                TC_B5=0.0
                TC_B6=0.0
                TC_B7=0.0
                TC_C1=0.0
                TC_C2=0.0
                TC_C3=0.0
                TC_C4=0.0
                TC_C5=0.0
                TC_C6=0.0
                TC_C7=0.0
                TC_D1=0.0
                TC_D2=0.0
                TC_D3=0.0
                TC_D4=0.0
                TC_D5=0.0
                TC_D6=0.0
                TC_D7=0.0
                TC_E1=0.0
                TC_E2=0.0
                TC_E3=0.0
                TC_E4=0.0
                TC_E5=0.0
                TC_E6=0.0
                TC_E7=0.0
                TC_F1=0.0
                TC_F2=0.0
                TC_F3=0.0
                TC_F4=0.0
                TC_F5=0.0
                TC_F6=0.0
                TC_F7=0.0
                TC_G1=0.0
                TC_G2=0.0
                TC_G3=0.0
                TC_G4=0.0
                TC_G5=0.0
                TC_G6=0.0
                TC_G7=0.0
                ETAPROD=0
                ETAREACT=0
                
                DO JJ=1,NS2
                        CONDSPEC = SPEC(JJ)
                        JJD = JJ
                        IF (CONDSPEC.EQ.'J') JJD = JJ - 1
                        IF (TD.GE.1.D+3) THEN
                        IF (REACT(J,1).EQ.JJ) THEN
                                TC_A1=TC(1,JJD)
                                TC_A2=TC(2,JJD)
                                TC_A3=TC(3,JJD)
                                TC_A4=TC(4,JJD)
                                TC_A5=TC(5,JJD)
                                TC_A6=TC(6,JJD)
                                TC_A7=TC(7,JJD)
                                IF(JJ.LT.NS2) ETAREACT=ETAREACT+1
                        ENDIF
                        IF (REACT(J,2).EQ.JJ) THEN 
                                TC_B1=TC(1,JJD)
                                TC_B2=TC(2,JJD)
                                TC_B3=TC(3,JJD)
                                TC_B4=TC(4,JJD)
                                TC_B5=TC(5,JJD)
                                TC_B6=TC(6,JJD)
                                TC_B7=TC(7,JJD)
                                IF(JJ.LT.NS2) ETAREACT=ETAREACT+1
                        ENDIF
                        IF (REACT(J,3).EQ.JJ) THEN 
                                TC_C1=TC(1,JJD)
                                TC_C2=TC(2,JJD)
                                TC_C3=TC(3,JJD)
                                TC_C4=TC(4,JJD)
                                TC_C5=TC(5,JJD)
                                TC_C6=TC(6,JJD)
                                TC_C7=TC(7,JJD)
                                IF(JJ.LT.NS2) ETAREACT=ETAREACT+1
                        ENDIF
                        IF (REACT(J,4).EQ.JJ) THEN 
                                TC_D1=TC(1,JJD)
                                TC_D2=TC(2,JJD)
                                TC_D3=TC(3,JJD)
                                TC_D4=TC(4,JJD)
                                TC_D5=TC(5,JJD)
                                TC_D6=TC(6,JJD)
                                TC_D7=TC(7,JJD)
                                IF(JJ.LT.NS2) ETAPROD=ETAPROD+1
                        ENDIF
                        IF (REACT(J,5).EQ.JJ) THEN 
                                TC_E1=TC(1,JJD)
                                TC_E2=TC(2,JJD)
                                TC_E3=TC(3,JJD)
                                TC_E4=TC(4,JJD)
                                TC_E5=TC(5,JJD)
                                TC_E6=TC(6,JJD)
                                TC_E7=TC(7,JJD)
                                IF(JJ.LT.NS2) ETAPROD=ETAPROD+1
                        ENDIF
                        IF (REACT(J,6).EQ.JJ) THEN 
                                TC_F1=TC(1,JJD)
                                TC_F2=TC(2,JJD)
                                TC_F3=TC(3,JJD)
                                TC_F4=TC(4,JJD)
                                TC_F5=TC(5,JJD)
                                TC_F6=TC(6,JJD)
                                TC_F7=TC(7,JJD)
                                IF(JJ.LT.NS2) ETAPROD=ETAPROD+1
                        ENDIF
                        IF (REACT(J,7).EQ.JJ) THEN 
                                TC_G1=TC(1,JJD)
                                TC_G2=TC(2,JJD)
                                TC_G3=TC(3,JJD)
                                TC_G4=TC(4,JJD)
                                TC_G5=TC(5,JJD)
                                TC_G6=TC(6,JJD)
                                TC_G7=TC(7,JJD)
                                IF(JJ.LT.NS2) ETAPROD=ETAPROD+1
                        ENDIF
                        ENDIF
                        IF (TD.LT.1.D+3) THEN
                        IF (REACT(J,1).EQ.JJ) THEN
                                TC_A1=TC(8,JJD)
                                TC_A2=TC(9,JJD)
                                TC_A3=TC(10,JJD)
                                TC_A4=TC(11,JJD)
                                TC_A5=TC(12,JJD)
                                TC_A6=TC(13,JJD)
                                TC_A7=TC(14,JJD)
                                IF(JJ.LT.NS2) ETAREACT=ETAREACT+1
                        ENDIF
                        IF (REACT(J,2).EQ.JJ) THEN 
                                TC_B1=TC(8,JJD)
                                TC_B2=TC(9,JJD)
                                TC_B3=TC(10,JJD)
                                TC_B4=TC(11,JJD)
                                TC_B5=TC(12,JJD)
                                TC_B6=TC(13,JJD)
                                TC_B7=TC(14,JJD)
                                IF(JJ.LT.NS2) ETAREACT=ETAREACT+1
                        ENDIF
                        IF (REACT(J,3).EQ.JJ) THEN 
                                TC_C1=TC(8,JJD)
                                TC_C2=TC(9,JJD)
                                TC_C3=TC(10,JJD)
                                TC_C4=TC(11,JJD)
                                TC_C5=TC(12,JJD)
                                TC_C6=TC(13,JJD)
                                TC_C7=TC(14,JJD)
                                IF(JJ.LT.NS2) ETAREACT=ETAREACT+1
                        ENDIF
                        IF (REACT(J,4).EQ.JJ) THEN 
                                TC_D1=TC(8,JJD)
                                TC_D2=TC(9,JJD)
                                TC_D3=TC(10,JJD)
                                TC_D4=TC(11,JJD)
                                TC_D5=TC(12,JJD)
                                TC_D6=TC(13,JJD)
                                TC_D7=TC(14,JJD)
                                IF(JJ.LT.NS2) ETAPROD=ETAPROD+1
                        ENDIF
                        IF (REACT(J,5).EQ.JJ) THEN 
                                TC_E1=TC(8,JJD)
                                TC_E2=TC(9,JJD)
                                TC_E3=TC(10,JJD)
                                TC_E4=TC(11,JJD)
                                TC_E5=TC(12,JJD)
                                TC_E6=TC(13,JJD)
                                TC_E7=TC(14,JJD)
                                IF(JJ.LT.NS2) ETAPROD=ETAPROD+1
                        ENDIF
                        IF (REACT(J,6).EQ.JJ) THEN 
                                TC_F1=TC(8,JJD)
                                TC_F2=TC(9,JJD)
                                TC_F3=TC(10,JJD)
                                TC_F4=TC(11,JJD)
                                TC_F5=TC(12,JJD)
                                TC_F6=TC(13,JJD)
                                TC_F7=TC(14,JJD)
                                IF(JJ.LT.NS2) ETAPROD=ETAPROD+1
                        ENDIF
                        IF (REACT(J,7).EQ.JJ) THEN 
                                TC_G1=TC(8,JJD)
                                TC_G2=TC(9,JJD)
                                TC_G3=TC(10,JJD)
                                TC_G4=TC(11,JJD)
                                TC_G5=TC(12,JJD)
                                TC_G6=TC(13,JJD)
                                TC_G7=TC(14,JJD)
                                IF(JJ.LT.NS2) ETAPROD=ETAPROD+1
                        ENDIF
                        ENDIF
                ENDDO
                
                BARFLAG = 0
                
                A1_REACT=TC_A1+TC_B1+TC_C1
                A2_REACT=TC_A2+TC_B2+TC_C2
                A3_REACT=TC_A3+TC_B3+TC_C3
                A4_REACT=TC_A4+TC_B4+TC_C4
                A5_REACT=TC_A5+TC_B5+TC_C5
                A6_REACT=TC_A6+TC_B6+TC_C6
                A7_REACT=TC_A7+TC_B7+TC_C7
                
                A1_PROD=TC_D1+TC_E1+TC_F1+TC_G1
                A2_PROD=TC_D2+TC_E2+TC_F2+TC_G2
                A3_PROD=TC_D3+TC_E3+TC_F3+TC_G3
                A4_PROD=TC_D4+TC_E4+TC_F4+TC_G4
                A5_PROD=TC_D5+TC_E5+TC_F5+TC_G5
                A6_PROD=TC_D6+TC_E6+TC_F6+TC_G6
                A7_PROD=TC_D7+TC_E7+TC_F7+TC_G7
                
                GFORM_REACT = A1_REACT*(1.0-LOG(TDP))
     &               -A2_REACT*5.0D-1*TDP-A3_REACT*1.667D-1*TDP**2.0
     &               -A4_REACT*8.333D-2*TDP**3.0
     &               -A5_REACT*5.0D-2*TDP**4.0+A6_REACT/TDP-A7_REACT

                GFORM_PROD = A1_PROD*(1.0-LOG(TDP))-A2_PROD*5.0D-1*TDP
     &               -A3_PROD*1.667D-1*TDP**2.0
     &               -A4_PROD*8.333D-2*TDP**3.0
     &               -A5_PROD*5.0D-2*TDP**4.0+A6_PROD/TDP-A7_PROD

                GREACTION = GFORM_PROD - GFORM_REACT

c		DECOMPOSITION REACTIONS
                IF ((ITYPE(J).EQ.0).AND.(MOD(J,2).EQ.1)) THEN
                  IF ((REACT(J,2).EQ.NS2).AND.(REACT(J,3).EQ.NS2)) THEN
                    BARFLAG = 1
                  ENDIF

                  KZERO=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                  KINF=A(J+1)*(TT**B(J+1))*EXP(-C(J+1)/TD)
                  K(J)=1.0D+0*KZERO*NHTOT/(1.0D+0 + KZERO/KINF*NHTOT)
                  IF(KZERO.EQ.0.D+0) K(J) = 0.D+0
                  IF(GREACTION.GE.C(J)/TD) 
     &               K(J)=K(J)*EXP(-GREACTION+C(J)/TD)
                  IF(BARFLAG.EQ.1) K(J)=K(J)*EXP(-6.0D+2/TD)
                ENDIF

c		ION-NEUTRAL FORWARD REACTIONS
                IF (ITYPE(J).EQ.2) THEN 
                        K(J)=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                        IF(GREACTION.GE.C(J)/TD)
     &                    K(J)=K(J)*EXP(-GREACTION+C(J)/TD)
                ENDIF

c		ION-NEUTRAL REVERSE REACTIONS
                IF (ITYPE(J).EQ.3) THEN
                  K(J)=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                  K(J)=K(J)*(1.38065D-22*TD)**(-ETAPROD+ETAREACT)
                  IF((-GREACTION.GE.C(J)/TD).AND.(K(J).GT.0.D+0)) 
     &              K(J)=K(J)*EXP(C(J)/TD)
                  IF((-GREACTION.LT.C(J)/TD).AND.(K(J).GT.0.D+0)) 
     &              K(J)=K(J)*EXP(-GREACTION)
                ENDIF

c		3-BODY ELECTRON-ION RECOMBINATION RATE
                IF ((ITYPE(J).EQ.5).AND.(MOD(J,2).EQ.1)) THEN
                  KZERO=A(J)*((3.33D-3*TEL)**B(J))*EXP(-C(J)/TEL)
                  KINF=A(J+1)*((3.33D-3*TEL)**B(J+1))*EXP(-C(J+1)/TEL)
                  K(J)=KZERO*NHTOT/(1.0D+0 + KZERO/KINF*NHTOT)
                  ENDIF

c		THERMAL IONIZATION RATE
                IF ((ITYPE(J).EQ.6).AND.(MOD(J,2).EQ.1)) THEN
                        KZERO=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                        KINF=A(J+1)*(TT**B(J+1))*EXP(-C(J+1)/TD)
                        K(J)=KZERO*NHTOT/(1.0D+0 + KZERO/KINF*NHTOT)
                        K(J)=K(J)*EXP(-GREACTION)
                        K(J)=K(J)*(1.38065D-22*TD)**(-ETAPROD+ETAREACT)
                ENDIF

c		NEUTRAL-NEUTRAL FORWARD REACTIONS	
                IF (ITYPE(J).EQ.7) THEN 
                  K(J)=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                  IF(GREACTION.GE.C(J)/TD)
     &              K(J)=K(J)*EXP(-GREACTION+C(J)/TD)
                ENDIF

c		(IRREVERSED) RADIATIVE ASSOCIATION                  
                IF (ITYPE(J).EQ.8) THEN 
                  K(J)=A(J)*((3.33D-3*TD)**B(J))*EXP(-C(J)/TD)
     &                 *1.D-10/(1.D-10+
     &                 NHTOT*6.4D-17*TD**(-3.1D+0)*EXP(-1.8D+2/TD))
                ENDIF

c		(IRREVERSED) DISSOCIATIVE ELECTRON-ION RECOMBINATION
                IF (ITYPE(J).EQ.10) THEN 
                  K(J)=A(J)*((3.33D-3*TEL)**B(J))*EXP(-C(J)/TEL)
                ENDIF

c		PHOTO-REACTIONS		
                IF (ITYPE(J).EQ.13) THEN
                  RADNUMNOW = RADNUM(J)
                  IF (PHOTOMODE.EQ."Y")
     &              CALL PHOTOCHEM(SIGMA,FIELD,RADNUMNOW,PHOTORATE,
     &                             ROTATION)
                  IF (PHOTOMODE.EQ."N") PHOTORATE = 0.D+0
                K(J)=PHOTORATE
c		ACCELERATE PHOTORATE BY 10, TURN OFF IN ALMOST ALL CASES!!!
c                K(J)=PHOTORATE*1.D+1
                ENDIF

c		2-BODY REVERSE NEUTRAL-NEUTRAL REACTIONS
                IF (ITYPE(J).EQ.14) THEN
                  K(J)=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                  IF ((B(J).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &              K(J)=A(J)*EXP(-C(J)/TD)
                  IF (-GREACTION.GT.C(J)/TD) K(J)=K(J)*EXP(C(J)/TD)
                  IF (-GREACTION.LE.C(J)/TD) K(J)=K(J)*EXP(-GREACTION)
                  K(J)=K(J)*(1.38065D-22*TD)**(-ETAPROD+ETAREACT)
                ENDIF

c		3-BODY NEUTRAL-NEUTRAL REACTIONS
                IF ((ITYPE(J).EQ.15).AND.(MOD(J,2).EQ.1)) THEN

                  IF ((REACT(J,5).EQ.NS2).AND.(REACT(J,6).EQ.NS2)
     &              .AND.(REACT(J,7).EQ.NS2)) THEN
                      BARFLAG = 1
                  ENDIF

                  KZERO=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                  IF ((B(J).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &              KZERO=A(J)*EXP(-C(J)/TD)
                  KINF=A(J+1)*(TT**B(J+1))*EXP(-C(J+1)/TD)
                  IF ((B(J+1).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &              KINF=A(J+1)*EXP(-C(J+1)/TD)
                  K(J)=KZERO*NHTOT/(1.0D+0 + KZERO/KINF*NHTOT)
                  IF(-GREACTION.GE.C(J)/TD) K(J)=K(J)*EXP(C(J)/TD)
                  IF(-GREACTION.LT.C(J)/TD) K(J)=K(J)*EXP(-GREACTION)
                  IF((-GREACTION.GT.6.0D+2)
     &              .AND.(-GREACTION.LT.C(J)/TD)) THEN
                    KZERO = A(J)*(TT**B(J))
                    IF ((B(J).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &                KZERO=A(J)
                    KINF=A(J+1)*(TT**B(J+1))
                    IF ((B(J+1).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &                KINF=A(J+1)
                    K(J)=KZERO*NHTOT/(1.0D+0 + KZERO/KINF*NHTOT)
                    NEWEXPFACTOR = -C(J)/TD-GREACTION
                    K(J)=K(J)*EXP(NEWEXPFACTOR)
                  ENDIF
                  IF((C(J)/TD.GT.6.0D+2)
     &              .AND.(-GREACTION.GT.C(J)/TD)) THEN
                    KZERO = A(J)*(TT**B(J))
                    IF ((B(J).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &                KZERO=A(J)
                    KINF=A(J+1)*(TT**B(J+1))
                    IF ((B(J+1).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &                KINF=A(J+1)
                    K(J)=KZERO*NHTOT/(1.0D+0 + KZERO/KINF*NHTOT)
                  ENDIF
                  K(J)=K(J)*(1.38065D-22*TD)**(-ETAPROD+ETAREACT)
                  IF (BARFLAG.EQ.1) THEN 
                    K(J)=K(J)*EXP(-6.0D+2/TD)
                  ENDIF
                ENDIF

c		3-BODY ION-NEUTRAL REVERSE REACTIONS
                IF ((ITYPE(J).EQ.16).AND.(MOD(J,2).EQ.1)) THEN
                  KZERO=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                  IF ((B(J).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &              KZERO=A(J)*EXP(-C(J)/TD)
                  KINF=A(J+1)*(TT**B(J+1))*EXP(-C(J+1)/TD)
                  IF ((B(J+1).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &              KINF=A(J+1)*EXP(-C(J+1)/TD)
                  K(J)=KZERO*NHTOT/(1.0D+0 + KZERO/KINF*NHTOT)
                  K(J)=K(J)*(1.38065D-22*TD)**(-ETAPROD+ETAREACT)
                  IF(-GREACTION.GE.C(J)/TD) K(J)=K(J)*EXP(C(J)/TD)
                  IF(-GREACTION.LT.C(J)/TD) K(J)=K(J)*EXP(-GREACTION)
                  IF((-GREACTION.GT.6.0D+2)
     &              .AND.(-GREACTION.LT.C(J)/TD)) THEN
                    KZERO = A(J)*(TT**B(J))
                    IF ((B(J).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &                KZERO=A(J)
                    KINF=A(J+1)*(TT**B(J+1))
                    IF ((B(J+1).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &                KINF=A(J+1)
                    K(J)=KZERO*NHTOT/(1.0D+0 + KZERO/KINF*NHTOT)
                    K(J)=K(J)*(1.38065D-22*TD)**(-ETAPROD+ETAREACT)
                    NEWEXPFACTOR = -C(J)/TD-GREACTION
                    K(J)=K(J)*EXP(NEWEXPFACTOR)
                  ENDIF
                  IF((C(J)/TD.GT.6.0D+2)
     &              .AND.(-GREACTION.GT.C(J)/TD)) THEN
                    KZERO = A(J)*(TT**B(J))
                    IF ((B(J).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &                KZERO=A(J)
                    KINF=A(J+1)*(TT**B(J+1))
                    IF ((B(J+1).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &                KINF=A(J+1)
                    K(J)=KZERO*NHTOT/(1.0D+0 + KZERO/KINF*NHTOT)
                    K(J)=K(J)*(1.38065D-22*TD)**(-ETAPROD+ETAREACT)
          ENDIF
          ENDIF

c		REVERSE 2-BODY REACTIONS WITH PRESSURE-DEPENDENT BRANCHING RATIOS
                IF ((ITYPE(J).EQ.20).AND.(MOD(J,2).EQ.1)) THEN

                  IF ((REACT(J,5).EQ.NS2).AND.(REACT(J,6).EQ.NS2)
     &              .AND.(REACT(J,7).EQ.NS2)) THEN
                      BARFLAG = 1
                  ENDIF

                  KZERO=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                  IF ((B(J).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &              KZERO=A(J)*EXP(-C(J)/TD)
                  KINF=A(J+1)*(TT**B(J+1))*EXP(-C(J+1)/TD)
                  IF ((B(J+1).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &              KINF=A(J+1)*EXP(-C(J+1)/TD)
                  K(J)=KINF/(1.0D+0 + KZERO/KINF*NHTOT)
                  K(J)=K(J)*(1.38065D-22*TD)**(-ETAPROD+ETAREACT)
                  IF(-GREACTION.GE.C(J)/TD) K(J)=K(J)*EXP(C(J)/TD)
                  IF(-GREACTION.LT.C(J)/TD) K(J)=K(J)*EXP(-GREACTION)
                  IF((-GREACTION.GT.6.0D+2)
     &              .AND.(-GREACTION.LT.C(J)/TD)) THEN
                    KZERO = A(J)*(TT**B(J))
                    IF ((B(J).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &                KZERO=A(J)
                    KINF=A(J+1)*(TT**B(J+1))
                    IF ((B(J+1).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &                KINF=A(J+1)
                    K(J)=KINF/(1.0D+0 + KZERO/KINF*NHTOT)
                    K(J)=K(J)*(1.38065D-22*TD)**(-ETAPROD+ETAREACT)
                    NEWEXPFACTOR = -C(J)/TD-GREACTION
                    K(J)=K(J)*EXP(NEWEXPFACTOR)
                  ENDIF
                  IF((C(J)/TD.GT.6.0D+2)
     &              .AND.(-GREACTION.GT.C(J)/TD)) THEN
                    KZERO = A(J)*(TT**B(J))
                    IF ((B(J).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &                KZERO=A(J)
                    KINF=A(J+1)*(TT**B(J+1))
                    IF ((B(J+1).LT.-1.D+0).AND.(TD.LT.3.0D+2))
     &                KINF=A(J+1)
                    K(J)=KINF/(1.0D+0 + KZERO/KINF*NHTOT)
                    K(J)=K(J)*(1.38065D-22*TD)**(-ETAPROD+ETAREACT)
                  ENDIF
                  IF (BARFLAG.EQ.1) THEN 
                    K(J)=K(J)*EXP(-6.0D+2/TD)
                  ENDIF
                ENDIF

c		2-BODY REACTIONS WITH PRESSURE-DEPENDENT BRANCHING RATIOS
                IF ((ITYPE(J).EQ.19).AND.(MOD(J,2).EQ.1)) THEN
                  IF ((REACT(J,2).EQ.NS2).AND.(REACT(J,3).EQ.NS2)) THEN
                    BARFLAG = 1
                  ENDIF

                  KZERO=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                  KINF=A(J+1)*(TT**B(J+1))*EXP(-C(J+1)/TD)
                  K(J)=KINF/(1.0D+0 + KZERO/KINF*NHTOT)
                  IF(KZERO.EQ.0.D+0) K(J) = 0.D+0
                  IF(GREACTION.GE.C(J)/TD) 
     &               K(J)=K(J)*EXP(-GREACTION+C(J)/TD)
                  IF(BARFLAG.EQ.1) K(J)=K(J)*EXP(-6.0D+2/TD)
                ENDIF

c		EVAPORATION & CONDENSATION

                TCOND = TD

c               EVAPORATION

                IF (ITYPE(J).EQ.17) THEN
                  TMIN = TC(9,REACT(J,1))
                  TMAX = TC(10,REACT(J,1))

                  IF (TCOND.LT.TMIN) TCOND = TMIN

                  PMIN = TC(11,REACT(J,1))
                  PVAP = TC(1,REACT(J,1)) + TC(2,REACT(J,1))
     &                  /(TCOND + TC(3,REACT(J,1)))
     &                  + TC(4,REACT(J,1))/TCOND**2.0
     &                  + TC(5,REACT(J,1))/TCOND**3.0
     &                  + TC(6,REACT(J,1))/TCOND**4.0
     &                  + TC(7,REACT(J,1))*LOG10(TCOND)
     &                  + TC(7,REACT(J,1))*LOG10(TCOND)
     &                  + TC(8,REACT(J,1))*LOG10(1.D+3/TCOND)
                  IF (PVAP.GT.4.D+0) PVAP = 4.D+0
                  PVAP = 1.0D+1**(PVAP)
                  NVAP = 7.243D+21*PVAP/TCOND
                  PGAS = NHTOT*TCOND*1.3806D-22
                  NCOND = SN(REACT(J,4))
                  NJCOND = SN(REACT(J,1))
                  NTOT = NCOND + NJCOND
                  IF(PGAS.LT.PMIN) K(J) = 0.D+0
                  IF(TCOND.GT.TMAX) K(J) = 0.D+0

                  K(J) = 0.D+0
                  KCOND = 4.55D-5*SQRT(TCOND/A(J))

                  K(J)=KCOND
                  IF((NVAP/NHTOT).LT.1.D-20) K(J)=0.D+0
                ENDIF

c               CONDENSATION

                IF (ITYPE(J).EQ.18) THEN
                  TMIN = TC(9,REACT(J,4))
                  TMAX = TC(10,REACT(J,4))

                  IF (TCOND.LT.TMIN) TCOND = TMIN

                  PMIN = TC(11,REACT(J,4))
                  PVAP = TC(1,REACT(J,4)) + TC(2,REACT(J,4))
     &                  /(TCOND + TC(3,REACT(J,4)))
     &                  + TC(4,REACT(J,4))/TCOND**2.0
     &                  + TC(5,REACT(J,4))/TCOND**3.0
     &                  + TC(6,REACT(J,4))/TCOND**4.0
     &                  + TC(7,REACT(J,4))*LOG10(TCOND)
     &                  + TC(7,REACT(J,4))*LOG10(TCOND)
     &                  + TC(8,REACT(J,4))*LOG10(1.D+3/TCOND)
                  IF (PVAP.GT.4.D+0) PVAP = 4.D+0
                  PVAP = 1.0D+1**(PVAP)
                  NVAP = 7.243D+21*PVAP/TCOND
                  PGAS = NHTOT*TCOND*1.3806D-22
                  NCOND = SN(REACT(J,4))
                  NJCOND = SN(REACT(J,1))
                  NTOT = NCOND + NJCOND
                  IF(PGAS.LT.PMIN) K(J) = 0.D+0
                  IF(TCOND.GT.TMAX) K(J) = 0.D+0

                  K(J) = 0.D+0
                  KCOND = 4.55D-5*SQRT(TCOND/A(J))

                  IF((CONDENSE.EQ.'Y').AND.(TCOND.LT.TMAX)) THEN
                    IF(NTOT.GT.NVAP) THEN
                       K(J)=KCOND*(NTOT/NVAP-1.D+0)
                       IF((NVAP/NHTOT).LT.1.D-20) K(J)=1.D+20*KCOND
                    ENDIF
                  ENDIF

                ENDIF

c		ARTIFICIAL WAY TO GET RID OF TERMINATING SPECIES

c		ARTIFICIAL WAY TO GET RID OF TERMINATING SPECIES
                 IF (ITYPE(J).EQ.88) THEN
                   K(J) = A(J)
                   K(J) = 0.D+0
                 ENDIF

c               OUTGASSING

                IF (ITYPE(J).EQ.42) THEN
                  PHI = A(J)
                  RESFRAC = SN(REACT(J,1))
                  LIFENOW = LIFE(REACT(J,1)-NS+NRES+1)
                  KOUTGAS = 0.D+0
                  IF ((ZNOW.LT.1.0D-1).AND.(DIRECTION.EQ.'UP'))
     &                CALL OUTGASSING(PHI,RESFRAC,ZPREV,ZNEXT,VZ,
     &                                LIFENOW,KOUTGAS)
                  K(J) = KOUTGAS
                ENDIF


c               DEPOSITION (NOT YET IMPLEMENTED!) itype 43

c               BANKING PARTICLES:
                IF (ITYPE(J).EQ.66) THEN
                  MASSJ = A(J)
                  SIGMAJ = B(J)
                  ALPHAT = C(J)
                  CALL MOLDIFF(DIRECTION,KBANK,KGO,MASSJ,MUAV,
     &                   SIGMAJ,SIGMAAV,LOGG,NHTOT,NHABOVE,NHBELOW,TD,
     &                   TDABOVE,TDBELOW,ZPREV,ZNEXT,ALPHAT,VZ,VZABOVE,
     &                   VZBELOW)
                  IF (DIFFMODE.EQ.'Y') K(J) = KBANK
                  IF (DIFFMODE.EQ.'N') K(J) = 0.D+0
                ENDIF

                IF (ITYPE(J).EQ.67) THEN
                  MASSJ = A(J)
                  SIGMAJ = B(J)
                  ALPHAT = C(J)
                  CALL MOLDIFF(DIRECTION,KBANK,KGO,MASSJ,MUAV,
     &                   SIGMAJ,SIGMAAV,LOGG,NHTOT,NHABOVE,NHBELOW,TD,
     &                   TDABOVE,TDBELOW,ZPREV,ZNEXT,ALPHAT,VZ,VZABOVE,
     &                   VZBELOW)
                  IF (DIFFMODE.EQ.'Y') K(J) = KGO
                  IF (DIFFMODE.EQ.'N') K(J) = 0.D+0
                ENDIF

c		NON-REVERSED 3-BODY REACTIONS
c		FOR ISOPRENE
                IF ((ITYPE(J).EQ.95).AND.(MOD(J,2).EQ.1)
     &              .AND.(ISISOPRENE.EQ.'Y')) THEN
                  KZERO=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                  KINF=A(J+1)*(TT**B(J+1))*EXP(-C(J+1)/TD)
                  K(J)=KZERO*NHTOT/(1.0D+0 + KZERO/KINF*NHTOT)
                ENDIF
                
                IF ((ITYPE(J).EQ.95).AND.(MOD(J,2).EQ.1)
     &              .AND.(ISISOPRENE.EQ.'Y')) THEN
                  K(J)=0.D+0
                ENDIF

c		FOR VENUS CHEMISTRY
                IF ((ITYPE(J).EQ.97).AND.(MOD(J,2).EQ.1)
     &              .AND.(ISVENUS.EQ.'Y')) THEN
                  KZERO=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                  KINF=A(J+1)*(TT**B(J+1))*EXP(-C(J+1)/TD)
                  K(J)=KZERO*NHTOT/(1.0D+0 + KZERO/KINF*NHTOT)
                ENDIF

                IF ((ITYPE(J).EQ.97).AND.(MOD(J,2).EQ.1)
     &              .AND.(ISVENUS.EQ.'N')) THEN
                  K(J)=0.D+0
                ENDIF


c               NON-REVERSED 2-BODY REACTIONS
c		FOR ISOPRENE
                IF ((ITYPE(J).EQ.96).AND.(ISISOPRENE.EQ.'Y')) THEN 
                  K(J)=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                ENDIF
               
                IF ((ITYPE(J).EQ.96).AND.(ISISOPRENE.EQ.'N')) THEN 
                  K(J)=0.D+0
                ENDIF
 
c		FOR VENUS CHEMISTRY
                IF ((ITYPE(J).EQ.98).AND.(ISVENUS.EQ.'Y')) THEN 
                  K(J)=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                ENDIF

                IF ((ITYPE(J).EQ.98).AND.(ISVENUS.EQ.'N')) THEN 
                  K(J)=0.D+0
                ENDIF

c		HYPOTHETICAL ION-NEUTRAL GLYCINE PRODUCTION
c		SHOULD BE TURNED OFF FOR MOST CHEMISTRY; REACTIONS ARE
c		SUSPECT!
                IF (ITYPE(J).EQ.99) THEN 
                  K(J)=A(J)*(TT**B(J))*EXP(-C(J)/TD)
                  K(J)=0.0D+00
                ENDIF

                  IF ((ITYPE(J).NE.13).AND.(ITYPE(J).NE.1).AND.
     &             (ITYPE(J).NE.10).AND.(ITYPE(J).NE.95).AND.
     &             (ITYPE(J).NE.96).AND.(ITYPE(J).NE.97).AND.
     &             (ITYPE(J).NE.98).AND.(ITYPE(J).NE.99).AND.
     &             (ITYPE(J).NE.17).AND.(ITYPE(J).NE.18).AND.
     &             (ITYPE(J).NE.42).AND.(ITYPE(J).NE.43).AND.
     &             (ITYPE(J).NE.66).AND.(ITYPE(J).NE.67)) THEN
                  IF ((TC_A6.EQ.0.0).AND.(REACT(J,1).LT.NS2)) 
     &              K(J)=0.0D+0
                  IF ((TC_B6.EQ.0.0).AND.(REACT(J,2).LT.NS2)) 
     &              K(J)=0.0D+0
                  IF ((TC_C6.EQ.0.0).AND.(REACT(J,3).LT.NS2)) 
     &              K(J)=0.0D+0
                  IF ((TC_D6.EQ.0.0).AND.(REACT(J,4).LT.NS2)) 
     &              K(J)=0.0D+0
                  IF ((TC_E6.EQ.0.0).AND.(REACT(J,5).LT.NS2)) 
     &              K(J)=0.0D+0
                  IF ((TC_F6.EQ.0.0).AND.(REACT(J,6).LT.NS2)) 
     &              K(J)=0.0D+0
                  IF ((TC_G6.EQ.0.0).AND.(REACT(J,7).LT.NS2)) 
     &              K(J)=0.0D+0
                ENDIF

                IF ((ITYPE(J).EQ.0).AND.(MOD(J,2).EQ.0)) THEN
                          K(J)=1.0D-51
                ENDIF
                
                IF ((ITYPE(J).EQ.5).AND.(MOD(J,2).EQ.0)) THEN
                          K(J)=1.0D-51
                ENDIF
                
                IF ((ITYPE(J).EQ.6).AND.(MOD(J,2).EQ.0)) THEN
                          K(J)=1.0D-51
                ENDIF
                
                IF ((ITYPE(J).EQ.15).AND.(MOD(J,2).EQ.0)) THEN
                          K(J)=1.0D-51
                ENDIF
                
                IF ((ITYPE(J).EQ.16).AND.(MOD(J,2).EQ.0)) THEN
                          K(J)=1.0D-51
                ENDIF
                
                IF ((ITYPE(J).EQ.19).AND.(MOD(J,2).EQ.0)) THEN
                          K(J)=1.0D-51
                ENDIF
                
                IF ((ITYPE(J).EQ.20).AND.(MOD(J,2).EQ.0)) THEN
                          K(J)=1.0D-51
                ENDIF

           ENDIF


         IR1=REACT(J,1)
         IR2=REACT(J,2)
         IR3=REACT(J,3)

c        For Jupiter only! ::
c         IF (J.EQ.1239) THEN
c           KZERO = 1.04D-24*TD**-1.5D+0
c           K(J) = 1.07D-10 
c     &           - 1.07D-10*KZERO*NHTOT/(1.07D-10 + KZERO*NHTOT)
c         ENDIF

c        For Earth only! ::
c          IF (J.EQ.1308) K(J) = K(J)*1.D+0
c          IF (J.EQ.1898) K(J) = K(J)*5.D-1
c          IF (J.EQ.65) K(J) = K(J)*1.D+0
c         IF (J.EQ.1506) K(J) = K(J)*DEXP(2.D+2/TD)
c         IF (J.EQ.2218) K(J) = K(J)*1.D+2
c         IF (J.EQ.1308) WRITE(*,*) K(J)


c        For Moses Code only! ::
c 
c         IF((J.EQ.1).AND.(TD.LT.2.58D+2)) K(J)=1.D-32
c         
c         IF(J.EQ.41) THEN
c           KZERO = 1.932D+3*TD**(-9.88D+0)*EXP(-7.544D+3/TD)
c     &             +5.109D-11*TD**(-6.25D+0)*EXP(-1.433D+3/TD)
c           KINF = 1.031D-10*TD**(-0.018D+0)*EXP(1.674D+1/TD)
c           FACTORC = 0.1855D+0*EXP(-TD/1.558D+2)
c     &              +0.8145D+0*EXP(-TD/1.645D+3)
c     &              +EXP(-4.531D+3/TD)
c           FACTORN = 0.75D+0 - 1.27D+0*LOG10(FACTORC)
c           FACTORF = LOG10(FACTORC)/(1.D+0 + (LOG10(KZERO*NHTOT/KINF)
c     &               /FACTORN)**2.D+0)
c           K(J) = KZERO*KINF*FACTORF/(KINF + KZERO*NHTOT)
c         ENDIF
c 
c         IF((J.EQ.73).AND.(TD.LT.2.95D+2)) K(J)=1.58D-8*TD**(-0.9D+0)
c 
c         IF(J.EQ.111) THEN
c           KZERO = 3.417D-8*TD**(-1.172D+0)*EXP(-1.32D+2/TD)
c     &             +1.535D-19*TD**(2.172D+0)*EXP(-5.478D+2/TD)
c           KINF = 7.976D-4*TD**(4.096D+0)*EXP(6.25D+2/TD)
c           FACTORC = 0.4863D+0*EXP(-TD/3.214D+2)
c     &              +0.5137D+0*EXP(-TD/3.D4)
c     &              +EXP(-2.804D+3/TD)
c           FACTORN = 0.75D+0 - 1.27D+0*LOG10(FACTORC)
c           FACTORF = LOG10(FACTORC)/(1.D+0 + (LOG10(KZERO*NHTOT/KINF)
c     &               /FACTORN)**2.D+0)
c           K(J) = KZERO*KINF*FACTORF/(KINF + KZERO*NHTOT)
c         ENDIF
c 
c         IF(J.EQ.112) THEN
c           KZERO = 1.092D-14*TD**(0.996D+0)*EXP(-1.606D+3/TD)
c           KINF = 5.864D-6*TD**(5.009D+0)*EXP(-9.494D+2/TD)
c           FACTORC = 0.8622D+0*EXP(-TD/9.321D+3)
c     &              +0.1378D+0*EXP(-TD/3.618D+2)
c     &              +EXP(-3.125D+3/TD)
c           FACTORN = 0.75D+0 - 1.27D+0*LOG10(FACTORC)
c           FACTORF = LOG10(FACTORC)/(1.D+0 + (LOG10(KZERO*NHTOT/KINF)
c     &               /FACTORN)**2.D+0)
c           K(J) = KZERO*KINF*FACTORF/(KINF + KZERO*NHTOT)
c         ENDIF
c 
c         IF(J.EQ.113) THEN
c           KZERO = 2.359D-10*TD**(-1.234D+0)*EXP(1.957D+1/TD)
c     &             +1.535D3*TD**(-4.484D+0)*EXP(-9.188D+3/TD)
c           KINF = 7.976D-14*TD**(6.721D+0)*EXP(1.521D+3/TD)
c           FACTORC = 0.5*EXP(-TD/2.2122D+4)
c     &              +0.5*EXP(-TD/1.749D+2)
c     &              +EXP(-3.047D+3/TD)
c           FACTORN = 0.75D+0 - 1.27D+0*LOG10(FACTORC)
c           FACTORF = LOG10(FACTORC)/(1.D+0 + (LOG10(KZERO*NHTOT/KINF)
c     &               /FACTORN)**2.D+0)
c           K(J) = KZERO*KINF*FACTORF/(KINF + KZERO*NHTOT)
c         ENDIF
c 
c         IF((J.EQ.183).AND.(TD.LT.2.5D+2)) K(J)=1.71D-12
c         IF((J.EQ.184).AND.(TD.LT.3.D+2)) K(J)=1.499D-13
c         IF((J.EQ.195).AND.(TD.LT.1.5D+2)) K(J)=1.2D-10
c         IF((J.EQ.210).AND.(TD.LT.2.97D+2)) K(J)=3.77D-10


c        For hot H2 ONLY!:
c         IF (J.EQ.1257) THEN
c           TCOLL = 20.0
c           NCOLL = 1.45D-16*NHTOT*TCOLL
c           NCOUNT = 0.D+0
c           IF(NCOLL.LT.0.5)THEN
c             TEMPCOLL = 1.58D+5
c           ENDIF
c           IF((NCOLL.GE.0.5).AND.(NCOLL.LT.1.D+1))THEN
c             TEMPCOLL = 1.58D+5
c             TEMPCOLL = TEMPCOLL*2.D+0**(-NCOLL)
c             DO WHILE (NCOUNT.LT.NCOLL)
c               TEMPCOLL = TEMPCOLL + TD*2.D+0**(-NCOUNT-1.D+0)
c               NCOUNT = NCOUNT + 1.0D+0
c             END DO
c           ENDIF
c           IF(NCOLL.GE.1.D+1) TEMPCOLL = TD
c           K(J) = A(J)*(TEMPCOLL/3.0D+2)**B(J)
c           K(J) = K(J)*DEXP(-C(J)/TEMPCOLL)
c         ENDIF

c	FOR VENUS ONLY!

c        IF (J.EQ.953) THEN
c            KZERO = 4.4D-31
c            KINF = 1.D-11
c            K(J)=KZERO*NHTOT/(1.0D+0 + KZERO/KINF*NHTOT)
c            K(J) = 1.D-28*EXP(6.D+3/TD)/K(J)
c        ENDIF

c        IF (J.EQ.5333) K(J) = 5.D-1*K(J)

c        IF (J.EQ.6021) WRITE(*,*) J, K(J)
c        IF (J.EQ.6194) WRITE(*,*) J, K(J)
c        IF (J.EQ.5758) WRITE(*,*) J, K(J)

c        IF (K(J).GT.1.D+0) WRITE(*,*) J, K(J)

        IF (K(J).NE.K(J)) K(J) = 0.D+0

        ENDDO        
c        STOP

C	modification of the rates for the uncertainties 
        IF (RANDMODE.EQ.'Y') THEN
        DO J=1,NRBIS2 
          K(J)=10**(DLOG10(K(J))+0.5*DLOG10(UNC(J))*RANDOM1(J))
        ENDDO
        ENDIF

C	remove the very small rate coefficients 
        DO J=1,NRBIS2
          IF (K(J).LT.1.D-50) K(J)=0.D-50
        ENDDO

C       writing the rate coefficients in the Kout.dat file 
        IF (JSTEP.EQ.0) THEN
          WRITE (9,*) 'Rate coefficients of the reactions'
          WRITE (9,*) 'Tk =    ',TD
          DO J=1,NRBIS2
            WRITE (9,*) J, K(J)
          ENDDO
        ENDIF

        RETURN
        END 

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c	this subroutine contains the differential equations
c
        SUBROUTINE F (NEQ, T, Y, YDOT)

        IMPLICIT NONE

        INTEGER NEQ, NRBIS2
        PARAMETER (NRBIS2=6612+5)

        INTEGER IR1,IR2,IR3,DES1,DES2,IPROD1,IPROD2,IPROD3,IPROD4,I,J

        DOUBLE PRECISION Y, YDOT, UP, DOWN, RATE,T
        DIMENSION Y(NEQ), YDOT(NEQ), UP(NEQ), DOWN(NEQ)

        INTEGER REACT(NRBIS2,7),NRBIS
        DOUBLE PRECISION K(NRBIS2)
        COMMON/EQUA/K,REACT,NRBIS

        IF (NRBIS.NE.NRBIS2) WRITE(*,*) 
     &   'Wrong number of reactions in subroutine f'


        DO I=1,NEQ
          UP(I)=0.d0
          DOWN(I)=0.d0
        ENDDO


C	the differential equations are calcultaed in a loop here 
        DO I=1,NRBIS2
 
                IR1=REACT(I,1)
                IR2=REACT(I,2)
                IR3=REACT(I,3)
                IPROD1=REACT(I,4)
                IPROD2=REACT(I,5)
                IPROD3=REACT(I,6)
                IPROD4=REACT(I,7)

                RATE=K(I)*Y(IR1)*Y(IR2)*Y(IR3)

                UP(IPROD1)=UP(IPROD1)+RATE
                UP(IPROD2)=UP(IPROD2)+RATE
                UP(IPROD3)=UP(IPROD3)+RATE
                UP(IPROD4)=UP(IPROD4)+RATE

                DOWN(IR1)=DOWN(IR1)+RATE
                DOWN(IR2)=DOWN(IR2)+RATE
                DOWN(IR3)=DOWN(IR3)+RATE

        ENDDO

        DO I=1,NEQ-1
          YDOT(I)=UP(I)-DOWN(I)
        ENDDO

        YDOT(NEQ)=0.d0

        RETURN
        END

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c	this subroutine computes the rates of formation and destruction 
C	for each species and put them in the increasing order 

        SUBROUTINE COMP_RATES (SN,SPEC,NEQ,LASTTIME)
 
        IMPLICIT NONE

        INTEGER J,NRBIS2,IR1,IR2,IR3,IPROD1,IPROD2,IPROD3,
     &          IPROD4,IY,M,I,NS2,NRBIS3,NEQ,NRES,LASTTIME,SWITCH,CONDSW
        PARAMETER (NRBIS2=6612+5,NS2=928+5+1,NRES=5)
        DOUBLE PRECISION SN,FORMTOT,DESTOT,RATE,X,FORM,DEST,INVHALFLIFE
        DIMENSION SN(NS2),FORMTOT(NS2),DESTOT(NS2),FORM(NS2,NRBIS2),
     &            DEST(NS2,NRBIS2),IY(NRBIS2),X(NRBIS2)
        CHARACTER CONDSPEC
        CHARACTER*8 SPEC(NS2)

        INTEGER REACT(NRBIS2,7),NRBIS
        DOUBLE PRECISION K(NRBIS2)
        COMMON/EQUA/K,REACT,NRBIS

        NRBIS3=NRBIS2
        IF (NS2.NE.NEQ) 
     &    WRITE (*,*) 'Wrong number of species in COMP_RATES' 

        DO J=1,NS2 
          FORMTOT(J)=0.D0
          DESTOT(J)=0.D0
        ENDDO

        DO I=1,NRBIS2
 
          IR1=REACT(I,1)
          IR2=REACT(I,2)
          IR3=REACT(I,3)
          IPROD1=REACT(I,4)
          IPROD2=REACT(I,5)
          IPROD3=REACT(I,6)
          IPROD4=REACT(I,7)

          RATE=K(I)*SN(IR1)*SN(IR2)*SN(IR3)

          FORM(IPROD1,I)=RATE
          FORM(IPROD2,I)=RATE
          FORM(IPROD3,I)=RATE
          FORM(IPROD4,I)=RATE
          DEST(IR1,I)=RATE
          DEST(IR2,I)=RATE
          DEST(IR3,I)=RATE

          FORMTOT(IPROD1)=FORMTOT(IPROD1)+RATE 
          FORMTOT(IPROD2)=FORMTOT(IPROD2)+RATE
          FORMTOT(IPROD3)=FORMTOT(IPROD3)+RATE 
          FORMTOT(IPROD4)=FORMTOT(IPROD4)+RATE 
          DESTOT(IR1)=DESTOT(IR1)+RATE
          DESTOT(IR2)=DESTOT(IR2)+RATE
          DESTOT(IR3)=DESTOT(IR3)+RATE

        ENDDO

C	put FORM and DEST in the increasing order 
        DO M=1,NS2-1
        DO I=1,NRBIS2
          X(I)=FORM(M,I)
          IY(I)=I
        ENDDO

        CALL SSORT (X, IY, NRBIS3)

        WRITE (11,*) SPEC(M)
        WRITE (11,*) 'REACTIONS OF PRODUTION',FORMTOT(M)
        DO I=1,5
          IF (X(I).NE.0D0) WRITE(11,*)  IY(I), X(I)
        ENDDO

        DO I=1,NRBIS2
          X(I)=DEST(M,I)
          IY(I)=I
        ENDDO

        CALL SSORT (X, IY, NRBIS3)

        WRITE (11,*) 'REACTIONS OF DESTRUCTION',DESTOT(M)
        DO I=1,5
          IF (X(I).NE.0D0) WRITE(11,*)  IY(I), -X(I)
        ENDDO

        ENDDO

        IF (LASTTIME.EQ.1) THEN
        DO J=1,NS2 
          FORMTOT(J)=0.D0
          DESTOT(J)=0.D0
        ENDDO

        SWITCH = 0

        DO I=1,NRBIS2
 
          IR1=REACT(I,1)
          IR2=REACT(I,2)
          IR3=REACT(I,3)
          IPROD1=REACT(I,4)
          IPROD2=REACT(I,5)
          IPROD3=REACT(I,6)
          IPROD4=REACT(I,7)
          
          IF (SPEC(IPROD1).EQ.'BC') SWITCH = 1

          CONDSW = 0
          
          CONDSPEC = SPEC(IR1)
          IF (CONDSPEC.EQ.'J') CONDSW = 1
          IF (CONDSPEC.EQ.'W') CONDSW = 1
          CONDSPEC = SPEC(IR2)
          IF (CONDSPEC.EQ.'J') CONDSW = 1
          IF (CONDSPEC.EQ.'W') CONDSW = 1
          CONDSPEC = SPEC(IR3)
          IF (CONDSPEC.EQ.'J') CONDSW = 1
          IF (CONDSPEC.EQ.'W') CONDSW = 1

          CONDSPEC = SPEC(IPROD1)
          IF (CONDSPEC.EQ.'W') CONDSW = 1
          CONDSPEC = SPEC(IPROD2)
          IF (CONDSPEC.EQ.'W') CONDSW = 1
          CONDSPEC = SPEC(IPROD3)
          IF (CONDSPEC.EQ.'W') CONDSW = 1
          CONDSPEC = SPEC(IPROD4)
          IF (CONDSPEC.EQ.'W') CONDSW = 1

C         ATTEMPTING TO FIX THE H2O LIFETIME PROBLEM WITH H2SO4. REMOVE LATER IF IT KEEPS CAUSING PROBLEMS.

          IF (SPEC(IR1).EQ.'H2SO4') CONDSW = 1
          IF (SPEC(IR2).EQ.'H2SO4') CONDSW = 1
          IF (SPEC(IR3).EQ.'H2SO4') CONDSW = 1
          IF (SPEC(IPROD1).EQ.'H2SO4') CONDSW = 1
          IF (SPEC(IPROD2).EQ.'H2SO4') CONDSW = 1
          IF (SPEC(IPROD3).EQ.'H2SO4') CONDSW = 1
          IF (SPEC(IPROD4).EQ.'H2SO4') CONDSW = 1
          IF (SPEC(IPROD1).EQ.'JH2SO4') CONDSW = 1
          IF (SPEC(IPROD2).EQ.'JH2SO4') CONDSW = 1
          IF (SPEC(IPROD3).EQ.'JH2SO4') CONDSW = 1
          IF (SPEC(IPROD4).EQ.'JH2SO4') CONDSW = 1

          IF (SPEC(IR1).EQ.'H2SO3') CONDSW = 1
          IF (SPEC(IR2).EQ.'H2SO3') CONDSW = 1
          IF (SPEC(IR3).EQ.'H2SO3') CONDSW = 1
          IF (SPEC(IPROD1).EQ.'H2SO3') CONDSW = 1
          IF (SPEC(IPROD2).EQ.'H2SO3') CONDSW = 1
          IF (SPEC(IPROD3).EQ.'H2SO3') CONDSW = 1
          IF (SPEC(IPROD4).EQ.'H2SO3') CONDSW = 1
          IF (SPEC(IPROD1).EQ.'JH2SO3') CONDSW = 1
          IF (SPEC(IPROD2).EQ.'JH2SO3') CONDSW = 1
          IF (SPEC(IPROD3).EQ.'JH2SO3') CONDSW = 1
          IF (SPEC(IPROD4).EQ.'JH2SO3') CONDSW = 1

          RATE=K(I)*SN(IR1)*SN(IR2)*SN(IR3)

          FORM(IPROD1,I)=RATE
          FORM(IPROD2,I)=RATE
          FORM(IPROD3,I)=RATE
          FORM(IPROD4,I)=RATE
          DEST(IR1,I)=RATE
          DEST(IR2,I)=RATE
          DEST(IR3,I)=RATE

          IF ((SWITCH.EQ.0).AND.(CONDSW.EQ.0)) THEN
          FORMTOT(IPROD1)=FORMTOT(IPROD1)+RATE 
          FORMTOT(IPROD2)=FORMTOT(IPROD2)+RATE
          FORMTOT(IPROD3)=FORMTOT(IPROD3)+RATE 
          FORMTOT(IPROD4)=FORMTOT(IPROD4)+RATE 
          DESTOT(IR1)=DESTOT(IR1)+RATE
          DESTOT(IR2)=DESTOT(IR2)+RATE
          DESTOT(IR3)=DESTOT(IR3)+RATE
          ENDIF

        ENDDO

        DO M=1,NS2-NRES-1

        IF (SN(M).GT.1.D-50) 
     &    INVHALFLIFE = (FORMTOT(M) - DESTOT(M))/SN(M)
        IF (SN(M).LE.1.D-50) 
     &    INVHALFLIFE = 0.D+0
        IF (ABS(INVHALFLIFE).LE.1.D-50) INVHALFLIFE = 0.D+0

        
c       Writing lifetimes to the Tout file.
        WRITE (17,*) SPEC(M), INVHALFLIFE

        ENDDO
        ENDIF

        RETURN
        END
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	this subroutine checks that the densities of charged species 
C	and elements are conservative 

        SUBROUTINE CHECKING (NS2,ELEMENT,NELEM,SN)
 
        IMPLICIT NONE
 
        INTEGER ELEMENT,NELEM,NS2,I,J
        DOUBLE PRECISION SN,PLUS,MINUS,DPM,CHECK_ELEM
        DIMENSION SN(NS2),ELEMENT(NELEM,NS2),CHECK_ELEM(NELEM-1)

        PLUS=0.D0
        MINUS=0.D0

        DO I=1,NELEM-1
          CHECK_ELEM(I)=0.D0
        ENDDO

        DO I=1,NS2-1
          IF (ELEMENT(1,I).EQ.1) PLUS=PLUS+SN(I)
          IF (ELEMENT(1,I).EQ.-1) MINUS=MINUS+SN(I)
        ENDDO

        DPM=(PLUS-MINUS)/PLUS

        WRITE (11,*) '(positive) - (negative)/(positive)',DPM
        WRITE (*,*) '(positive) - (negative)/(positive)',DPM

        DO I=1,NELEM-1
          DO J=1,NS2-3
            CHECK_ELEM(I)=CHECK_ELEM(I)+FLOAT(ELEMENT(I+1,J))*SN(J)
          ENDDO
        ENDDO

        WRITE(11,*) '( H  He C  N  O  Si S  Fe Na Mg Cl P  K  Ar Ti )'
        WRITE(11,31) (CHECK_ELEM(I),I=1,NELEM-1)
        WRITE(*,*) '( H  He C  N  O  Si S  Fe Na Mg Cl P  K  Ar Ti )'
        WRITE(*,31) (CHECK_ELEM(I),I=1,NELEM-1)

31      FORMAT(8('   ',E9.3))

        RETURN
        END

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c	dummy subroutine 
C	this subroutine is needed because we are not supplying 
C	the jacobian matrix 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        SUBROUTINE DUMMY 
 
        ENTRY JAC (NEQ, T, Y, ML, MU, PD, NROWPD)
 
        RETURN
        END

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	subroutine to sort the numbers

      SUBROUTINE SSORT (X, IY, N)
      IMPLICIT NONE

c
c    Example of an Insertion Sort
c
C***BEGIN PROLOGUE  SSORT
C***PURPOSE  Sort an array and make the same interchanges in
C            an auxiliary array.  The array is sorted in
C            decreasing order.
C***TYPE      SINGLE PRECISION
C***KEYWORDS  SORT, SORTING
C
C   Description of Parameters
C      X - array of values to be sorted   (usually abscissas)
C      IY - array to be carried with X (all swaps of X elements are
C          matched in IY .  After the sort IY(J) contains the original
C          postition of the value X(J) in the unsorted X array.
C      N - number of values in array X to be sorted
C      KFLAG - Not used in this implementation
C
C***REVISION HISTORY  (YYMMDD)
C   950310  DATE WRITTEN
C   John Mahaffy
C***END PROLOGUE  SSORT
C     .. Scalar Arguments ..
      INTEGER N
C     .. Array Arguments ..
      DOUBLE PRECISION X(*)
      INTEGER IY(*)
C     .. Local Scalars ..
      DOUBLE PRECISION TEMP
      INTEGER I, J, L, ITEMP
C     .. External Subroutines ..
C     None
C     .. Intrinsic Functions ..
C     None
C
C***FIRST EXECUTABLE STATEMENT  SSORT
C
      DO 100 I=2,N
         IF ( X(I).GT.X(I-1) ) THEN
            DO 50 J=I-2,1,-1
              IF(X(I).LT.X(J)) go to 70
  50          CONTINUE
            J=0
  70        TEMP=X(I)
            ITEMP=IY(I)
            DO 90 L=I,J+2,-1
              IY(L)=IY(L-1)
  90          X(L)=X(L-1)
            X(J+1)=TEMP
            IY(J+1)=ITEMP
         ENDIF
  100 CONTINUE
      RETURN
      END

      SUBROUTINE PHOTOREAD (SIGMA,FIELD)
      IMPLICIT NONE

      INTEGER WLNUM,PRNUM,I,J,K
      DOUBLE PRECISION SIGMA,FIELD
      CHARACTER*(12*(209+1)) LINE
      CHARACTER(LEN=5)FMT0
      CHARACTER(LEN=1)FMT1
      CHARACTER(LEN=2)FMT2
      CHARACTER(LEN=3)FMT3
      CHARACTER(LEN=4)FMT4
      PARAMETER (WLNUM=10001,PRNUM=209+1)
      DIMENSION SIGMA(PRNUM,WLNUM),FIELD(2,WLNUM)

      DO J=1,WLNUM
c        IF (J.GT.1) THEN
c          DO K=1,PRNUM
c            WRITE(FMT0,26) K
c            READ (LINE,FMT0) SIGMA(K,J)
c          ENDDO
c        ENDIF

        READ (13,30) LINE
c        WRITE(*,*) LINE
        DO K=1,PRNUM
          IF(J.GT.1) THEN
            WRITE(FMT0,26) K*12-12
            IF ((K.GT.1).AND.(K.LE.9)) READ(FMT0,22) FMT2
            IF ((K.GT.9).AND.(K.LE.84)) READ(FMT0,23) FMT3
            IF (K.GT.84) READ(FMT0,24) FMT4
            IF (K.EQ.1) READ (LINE,21) SIGMA(K,J)
            IF ((K.GT.1).AND.(K.LE.9)) READ (LINE,"("// FMT2 // 
     &        "X,E12.3)") SIGMA(K,J) 
            IF ((K.GT.9).AND.(K.LE.84)) READ (LINE,"("// FMT3 // 
     &        "X,E12.3)") SIGMA(K,J) 
            IF (K.GT.84) READ (LINE,"("// FMT4 // "X,E12.3)") 
     &        SIGMA(K,J) 
          ENDIF
        ENDDO
      ENDDO

      DO J=1,WLNUM
        IF (J.EQ.1) READ (14,*)
        IF (J.GT.1) READ (14,25) FIELD(1,J), FIELD(2,J)
      ENDDO
      
21    FORMAT (E12.3) 
22    FORMAT (3X,A2) 
23    FORMAT (2X,A3) 
24    FORMAT (1X,A4) 
25    FORMAT (2(E12.3))
26    FORMAT (I5)
30    FORMAT (A)

c     Test to make sure that the number and value of wavelength elements is 
c     the same for the field file as for the cross section file.

      DO J=1,WLNUM
        IF (SIGMA(1,J).NE.FIELD(1,J)) THEN
          WRITE (*,*) "ERROR! The field file and cross-section file
     &                 don't match! Fix one or the other!"
          WRITE (*,*) "Offending line! Sigma Wavelength = ",SIGMA(1,J)
          WRITE (*,*) "Offending line! Field Wavelength = ",FIELD(1,J)
          STOP
        ENDIF
      ENDDO

      RETURN
      END

      SUBROUTINE PHOTOCHEM (SIGMA,FIELD,RADNUMNOW,PHOTORATE,ROTATION)
      IMPLICIT NONE

      INTEGER NRBIS2,NS2,WLNUM,PRNUM,RADNUMNOW,J,K
      CHARACTER ROTATION
      DOUBLE PRECISION SIGMA,FIELD,FIELDAV,PHOTORATE,DRATE,DWL,CFACT
      PARAMETER (WLNUM=10001,PRNUM=209+1)
      DIMENSION SIGMA(PRNUM,WLNUM),FIELD(2,WLNUM)

      DOUBLE PRECISION MP, KB, PI, MOLE
      DATA MP,KB,PI,MOLE
     &     /1.67262D-24, 1.38064D-16, 3.14159265, 6.023D23/

      PHOTORATE = 0.0D+0
c      WRITE(*,*) "I just made my first subroutine in FORTRAN!"

      IF (ROTATION.EQ.'S') CFACT = 1.D+0
      IF (ROTATION.EQ.'T') CFACT = 2.D+0*PI
      IF (ROTATION.EQ.'F') CFACT = 4.D+0*PI

      DO J=1,WLNUM
        IF (J.LT.WLNUM) THEN
           DWL = FIELD(1,J+1)-FIELD(1,J)
           FIELDAV = (FIELD(2,J+1)+FIELD(2,J))/2.0
           FIELDAV = FIELD(2,J)
        ENDIF
        IF (J.EQ.WLNUM) THEN
           DWL = FIELD(1,J)-FIELD(1,J-1)
           FIELDAV = (FIELD(2,J)+FIELD(2,J-1))/2.0
           FIELDAV = FIELD(2,J)
        ENDIF
c       Rates for all reactions besides H2 -> H + H
        IF (RADNUMNOW.GT.0) DRATE = SIGMA(RADNUMNOW+1,J)*FIELDAV*DWL
c       Rate for H2 -> H+H, combines two sigma-values, columns 68 & 69
        IF (RADNUMNOW.EQ.-1) DRATE = (SIGMA(68+1,J)+SIGMA(69+1,J))
     &                               *FIELDAV*DWL
        PHOTORATE = PHOTORATE + DRATE
      ENDDO
      PHOTORATE = 2.0D+0*PI*PHOTORATE/CFACT
      RETURN
      END

      SUBROUTINE SETDIFF(ELEMENT,SN,MUAV,DIRECTION)

      IMPLICIT NONE

      INTEGER ELEMENT,THRESHOLD,NS,NRES,NELEM,J
      DOUBLE PRECISION SN,MASSJ,MUAV,MUAVNEW,DENTOT
      CHARACTER DIRECTION*2
      PARAMETER (NS=928+5+1,NRES=5)
      PARAMETER (NELEM=15+1,THRESHOLD=510)
      DIMENSION SN(NS), ELEMENT(NELEM,NS)

      MUAVNEW = 0.D+0
      DENTOT = 0.D+0
      
      DO J=0,THRESHOLD
        MASSJ = ELEMENT(2,J) + ELEMENT(3,J)*4.D+0 
     &          + ELEMENT(4,J)*1.2D+1 + ELEMENT(5,J)*1.4D+1
     &          + ELEMENT(6,J)*1.6D+1 + ELEMENT(7,J)*2.8D+1
     &          + ELEMENT(8,J)*3.2D+1 + ELEMENT(9,J)*5.6D+1
     &          + ELEMENT(10,J)*2.3D+1 + ELEMENT(11,J)*2.4D+1
     &          + ELEMENT(12,J)*3.5D+1 + ELEMENT(13,J)*3.1D+1
     &          + ELEMENT(14,J)*3.9D+1 + ELEMENT(15,J)*4.0D+1
     &          + ELEMENT(16,J)*4.8D+1
        MUAVNEW = MUAVNEW + MASSJ*SN(J)
        DENTOT = DENTOT + SN(J)
      ENDDO

      MUAVNEW = MUAVNEW/DENTOT

      MUAV = MUAVNEW

      DO J=THRESHOLD,NS-NRES-1
         MASSJ = ELEMENT(2,J) + ELEMENT(3,J)*4.D+0 
     &          + ELEMENT(4,J)*1.2D+1 + ELEMENT(5,J)*1.4D+1
     &          + ELEMENT(6,J)*1.6D+1 + ELEMENT(7,J)*2.8D+1
     &          + ELEMENT(8,J)*3.2D+1 + ELEMENT(9,J)*5.6D+1
     &          + ELEMENT(10,J)*2.3D+1 + ELEMENT(11,J)*2.4D+1
     &          + ELEMENT(12,J)*3.5D+1 + ELEMENT(13,J)*3.1D+1
     &          + ELEMENT(14,J)*3.9D+1 + ELEMENT(15,J)*4.0D+1
     &          + ELEMENT(16,J)*4.8D+1
        IF ((DIRECTION.EQ.'UP').AND.(MASSJ.GT.MUAV)) SN(J) = 0.D+0
        IF ((DIRECTION.EQ.'DN').AND.(MASSJ.LT.MUAV)) SN(J) = 0.D+0
      ENDDO


      RETURN
      END


      SUBROUTINE MOLDIFF(DIRECTION,KBANK,KGO,MASSJ,MUAV,SIGMAJ,
     &                   SIGMAAV,LOGG,NHTOT,NHABOVE,NHBELOW,TD,
     &                   TDABOVE,TDBELOW,ZPREV,ZNEXT,ALPHAT,VZ,VZABOVE,
     &                   VZBELOW)

      IMPLICIT NONE
      DOUBLE PRECISION NHTOT,NHABOVE,NHBELOW,TD,TDABOVE,TDBELOW,MASSJ,
     &                 MUAV,P,PABOVE,PBELOW,KB,SIGMAJ,SIGMAAV,SIGMA12,
     &                 MP,GRAV,KBANK,KGO,LOGG,OMEGA,TAVABOVE,TAVBELOW,
     &                 NHAVABOVE,NHAVBELOW,DZZ,DABOVE,DBELOW,DAVABOVE,
     &                 DAVBELOW,DIFFABOVE,DIFFBELOW,ZPREV,ZNEXT,ALPHAT,
     &                 VZ,VZABOVE,VZBELOW,KIFFABOVE,KIFFBELOW,ZSTEP
      CHARACTER DIRECTION*2

      KB = 1.3806D-16
      MP = 1.6726D-24
      GRAV = 1.0D+1**LOGG
      OMEGA = 1.0D+0

      ZSTEP = 5.D-1*(ZPREV+ZNEXT)

      TAVABOVE = 5.D-1*(TDABOVE + TD)
      TAVBELOW = 5.D-1*(TDBELOW + TD)
      NHAVABOVE = 5.D-1*(NHABOVE + NHTOT)
      NHAVBELOW = 5.D-1*(NHBELOW + NHTOT)

      P = NHTOT*KB*TD*1.D-6
      PABOVE = NHABOVE*KB*TDABOVE*1.D-6
      PBELOW = NHBELOW*KB*TDBELOW*1.D-6
      
      SIGMA12 = 0.5D+0*(SIGMAJ + SIGMAAV)

      DZZ = 1.858D-3*TD**(1.5D+0)
     &      *(1.0D+0/MASSJ + 1.0D+0/MUAV)**0.5D+0
     &      /(P*SIGMA12*SIGMA12*OMEGA)

      DABOVE = 1.858D-3*TDABOVE**(1.5D+0)
     &      *(1.0D+0/MASSJ + 1.0D+0/MUAV)**0.5D+0
     &      /(PABOVE*SIGMA12*SIGMA12*OMEGA)

      DBELOW = 1.858D-3*TDBELOW**(1.5D+0)
     &      *(1.0D+0/MASSJ + 1.0D+0/MUAV)**0.5D+0
     &      /(PBELOW*SIGMA12*SIGMA12*OMEGA)

      DZZ = 2.3D+17*TD**(0.765)/((NHTOT)**9.2D-1)
     &      *(16.04/MASSJ*((MASSJ + 2.016)/18.059))**0.5

      DABOVE = 2.3D+17*TDABOVE**(0.765)/((NHABOVE)**9.2D-1)
     &      *(16.04/MASSJ*((MASSJ + 2.016)/18.059))**0.5

      DBELOW = 2.3D+17*TDBELOW**(0.765)/((NHBELOW)**9.2D-1)
     &      *(16.04/MASSJ*((MASSJ + 2.016)/18.059))**0.5

      DAVABOVE = 5.D-1*(DABOVE + DZZ)
      DAVBELOW = 5.D-1*(DBELOW + DZZ)
      KIFFABOVE = 0.D+0
      KIFFBELOW = 0.D+0
c      IF (DIRECTION.EQ.'DN')
c     &  KIFFABOVE = 5.D-1*(VZABOVE + VZ)/ZSTEP + DAVABOVE/(ZSTEP*ZSTEP)
c      IF (DIRECTION.EQ.'ABOVE')
c     &  KIFFBELOW = 5.D-1*(VZBELOW + VZ)/ZSTEP + DAVBELOW/(ZSTEP*ZSTEP)

      DIFFABOVE = DAVABOVE/(2.D+0*ZSTEP*ZSTEP)*(
     &           ((MUAV - MASSJ)*MP*GRAV*ZSTEP/(KB*TAVABOVE))
     &          -(ALPHAT/TAVABOVE*(TDABOVE - TD)))

      DIFFBELOW = DAVBELOW/(2.D+0*ZSTEP*ZSTEP)*(
     &           ((MUAV - MASSJ)*MP*GRAV*ZSTEP/(KB*TAVBELOW))
     &          -(ALPHAT/TAVBELOW*(TDBELOW - TD)))

      KBANK = KIFFABOVE*NHAVABOVE/NHTOT + KIFFBELOW*NHAVBELOW/NHTOT +
     &        DIFFABOVE*NHAVABOVE/NHTOT - DIFFBELOW*NHAVBELOW/NHTOT

      IF (DIRECTION.EQ.'UP') 
     &        KGO = (KIFFBELOW+DIFFBELOW)*NHAVBELOW/NHBELOW
      IF (DIRECTION.EQ.'DN') 
     &        KGO = (KIFFABOVE-DIFFABOVE)*NHAVABOVE/NHABOVE

c     TESTING TO SEE IF THIS FIXES THE CHEMISTRY GOING DOWN. REMOVE LATER IF DOESNT WORK

      IF (DIRECTION.EQ.'DN') KGO = 0.D+0
      IF (DIRECTION.EQ.'DN') KBANK = 0.D+0

c      WRITE(*,*) "KGO = ",KGO
c      WRITE(*,*) "KBANK = ",KBANK

      IF (KGO.LT.0.D+0) KGO = 0.D+0
      IF (KBANK.LT.0.D+0) KBANK = 0.D+0

c      WRITE(*,*) "KGO = ",KGO
c      WRITE(*,*) "KBANK = ",KBANK

      RETURN
      END


      SUBROUTINE OUTGASSING(PHI,RESFRAC,ZPREV,ZNEXT,VZ,LIFENOW,KOUTGAS)

      DOUBLE PRECISION PHI,RESFRAC,ZPREV,ZNEXT,VZ,LIFENOW,KOUTGAS

      DOUBLE PRECISION RATE,DZ,TSCALE,CORRECT

      CORRECT = 1.D+0

      TSCALE = ZNEXT/VZ

      IF (LIFENOW.GT.TSCALE) CORRECT = LIFENOW/TSCALE

      DZ = 5.0D-1*(ZPREV + ZNEXT)

      RATE = 2.0D+0*PHI/DZ

      KOUTGAS = RATE/RESFRAC*CORRECT

      END
