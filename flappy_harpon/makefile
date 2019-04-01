CPP=gcc   #Commande du compilateur
CFLAGS=-O3 #Option d'optimisation du programme
# LDFLAGS=-lSDL2 -lSDL2_mixer #Linker
LDFLAGS=-lSDL -lSDL_mixer -lSDL_image -lSDL_ttf -lm#Linker

OBJS=play.o

all:
	cd lib_flappy; make; cd ..;
	make play

-include $(OBJS:.o=.d)

play: play.o lib_flappy/lib_flappy.a
	$(CPP) -g $(CFLAGS) -o play play.o lib_flappy/lib_flappy.a $(LDFLAGS)

%.o: %.c
	$(CPP) -g $(CFLAGS) -c $*.c
	$(CPP) -g -MM $*.c > $*.d

clean:
	rm -fr *.o *.d play test

mrproper: clean
	rm -fr play test
