OBJS := headposeimg.o
LIBS := -lvaal

%.o : %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) 

headposeimg: $(OBJS)
	dpkg -L libvaal
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS)


install: headposeimg
	mkdir -p $(WORKDIR)
	cp headposeimg $(WORKDIR)/


clean:
	rm -f *.o
	rm -f headposeimg
