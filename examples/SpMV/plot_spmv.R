library(Rmisc);
library(ggplot2);
library(data.table);

# initialize the data frame
df = data.frame(col1 = numeric(), col2 = character(), col2 = numeric(), stringsAsFactors = FALSE);

df[1, ] <- c(0.05, "coo", 454.4);
df[2, ] <- c(0.1, "coo", 715.037);
df[3, ] <- c(0.15, "coo", 993.799);
df[4, ] <- c(0.2, "coo", 1331.4);
df[5, ] <- c(0.25, "coo", 1571.84);
df[6, ] <- c(0.3, "coo", 1829.88);

df[7, ] <- c(0.05, "csr", 358.09);
df[8, ] <- c(0.1, "csr", 452.236);
df[9, ] <- c(0.15, "csr", 583.662);
df[10, ] <- c(0.2, "csr", 694.279);
df[11, ] <- c(0.25, "csr", 821.266);
df[12, ] <- c(0.3, "csr", 945.479);

df[13, ] <- c(0.05, "ell", 365.714);
df[14, ] <- c(0.1, "ell", 475.98);
df[15, ] <- c(0.15, "ell", 623.596);
df[16, ] <- c(0.2, "ell", 754.561);
df[17, ] <- c(0.25, "ell", 922.015);
df[18, ] <- c(0.3, "ell", 1041.86);

# set the names of the columns
colnames(df) <- c("density", "type", "latency");

# convert the columns in numeric data type
df$density <- as.numeric(as.character(df$density));
df$latency <- as.numeric(as.character(df$latency));

print(df)

# plotting
plot <- ggplot(df, aes(x = density, y = latency, color = type, group = type)) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5)) +
    xlab("density") +
    ylab("kernel latency (usec)") +
    geom_point(size=4.4) +
    geom_line(linewidth=0.8) +
    ggtitle("SpMV with different matrix layout") +
    theme(legend.position="bottom") +
    theme(legend.title= element_blank(), legend.text=element_text(size=14,face="bold")) +
    theme(axis.text=element_text(size=14), axis.title=element_text(size=16,face="bold"), plot.title=element_text(size=18,face="bold")) +
    theme(aspect.ratio=0.5);

# saving the plot in a pdf file
ggsave("spmv.pdf", plot);
