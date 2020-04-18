%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SMI,CCM-SMI,bind
% ����܂ł�propSMI,CCM�폜�o�[�W����
% 
% 
% 
% ���M�͂����̃V���{���~�L�����A�~���[�U
% 
% �������ɂ͊֐�
% signal_generater
% tx_prepare
% singlepath
% shift
% rx_prepare
% dem_mod
% demodulation
% ber_calculaton
% tocf
% ���K�v(ifft,fft�̓]�u�̂Ƃ��낪������Ɖ�����)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % ���̃t�H���g�T�C�Y
% set(0, 'defaultAxesFontSize', 16);
% set(0, 'defaultAxesFontName', 'Arial');
% % set(0, 'defaultAxesFontName', 'Times');
% 
% % �^�C�g���A���߂Ȃǂ̃t�H���g�T�C�Y
% set(0, 'defaultTextFontSize', 16);
% 
% % ���̐��̑���
% set(0, 'DefaultLineLineWidth', 1)
% set(0, 'DefaultAxesLineWidth', 1);

clear;
% close all;
tic;
rand('state',sum(100*clock)); %rand�֐��̏�����

%% �l�̒�`
antenna_form=2;                         %1->linear, 2->rectangular
number_ant_x=8;
number_ant_y=8;
number_ant=number_ant_x*number_ant_y;                           %�A���e�i��(x*y�ɂȂ�悤�ɂ��Ă���)
number_pilot=16;                         %�p�C���b�g��


Doppler=0.01;              %0.01               %�h�b�v���[���g��(10Hz)   �Ȃ���0.001�ȉ����ƈ����Ȃ�H0.01�܂ł͕��ʂɂ悭�Ȃ�

ebno_min=0;                             %�ŏ�Eb/No�l
ebno_max=15;                            %�ő�Eb/No�l
ebno_step=3;                            %Eb/No�Ԋu

% number_bind=2;

repeat=100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ���̉��͊�{������Ȃ�(���[�U��k��OK)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


guard_interval=16;                       %�K�[�h�C���^�[�o��
trans_rate=2*10^6;                      %�`���ш敝(2MHz)
number_symbol=20;                       %�V���{����
number_carrier=64;                      %�L��������
ifft_size=64;                           %IFFT�T�C�Y

D=1;                                    %Desired_User
% number_user=length(DOAvector);          %���[�U��(=�����p��)
deg2rad=pi/180;
d=0.5;                                  %�A���e�i�Ԋu

modulation_index=4;                     %�ϒ�����(4->16QAM)
path_index=3;                           %1->�V���O���p�X�C2->�}���`�p�X�C3->rice
number_path=5;
path_interval=1;
decay_factor=1;
K_dB=20;
K=10^(K_dB/10);
sigma=0.2;


%�������m��
% pilot=zeros(number_user,number_carrier,number_user);
% faded_signal=zeros(number_user,(number_carrier+guard_interval)*(number_symbol+number_pilot));
% channel=zeros(number_ant,number_user,number_carrier);
Rec_smi=zeros(number_carrier,number_symbol+number_pilot);
Rec_ccm=zeros(number_carrier,number_symbol+number_pilot);
Rec_bind=zeros(number_carrier,number_symbol+number_pilot);
Rec_zf=zeros(number_carrier,number_symbol+number_pilot);


for k=[16]
    
if k==1
    DOAvector=[40];
elseif k==2
%     DOAvector=[40,30];
    DOAvector=[40,-30];
elseif k==4
%     DOAvector=[40,30,-40,70];
    DOAvector=[40,-30,-70,80];
    DOAvector_x=[-30,40,-70,80];    %theta
    DOAvector_y=[0,-30,20,-50];    %phi
elseif k==8
%     DOAvector=[40,30,-40,70,10,-10,80,-60];
%     DOAvector=[40,-30,-70,80,60,20,-10,-50];
    DOAvector_x=[-30,40,-70,80,60,20,-10,-50];
    DOAvector_y=[0,-30,20,-50,60,10,-60,-10];
elseif k==16
    DOAvector_x=[-30,-80,-70,-60,-50,-40,-20,-10,0,10,20,30,40,50,60,70];
    DOAvector_y=[0,20,-40,50,-30,0,10,-10,-20,30,20,-30,40,-50,-20,10];
elseif k==32
    DOAvector_x=[-30,-80,-70,-60,-50,-40,-20,-10,0,10,20,30,40,50,60,70,-35,-85,-75,-65,-55,-45,-25,-15,5,15,25,35,45,55,65,75];
    DOAvector_y=[0,20,-40,50,-30,0,10,-10,-20,30,20,-30,40,-50,-20,10,5,25,-45,55,-35,5,15,-15,-25,35,25,-35,45,-55,-25,15];
end

number_user=length(DOAvector_x);

pilot=zeros(number_user,number_carrier,number_user);
%�p�C���b�g�M���쐬
hadamard_m=hadamard(number_user);
for nu=1:number_user
    pilot(:,:,nu)=repmat(hadamard_m(:,nu),1,ifft_size);
end

if k==1
    pilot_signal=ones(number_pilot,number_carrier,number_user);
elseif k==2
    if number_pilot==number_user
        pilot_signal=pilot;
    elseif number_pilot>number_user
        %number_pilot��user�ȏ�̂Ƃ�����
        pilot_signal=[repmat(pilot(1,:,:),number_pilot/2,1);repmat(pilot(2,:,:),number_pilot/2,1)];
    end
elseif k==4
    if number_pilot==number_user
        pilot_signal=pilot;
    elseif number_pilot>number_user
        %number_pilot��user�ȏ�̂Ƃ�����
        pilot_signal=[repmat(pilot(1,:,:),number_pilot/4,1);repmat(pilot(2,:,:),number_pilot/4,1);repmat(pilot(3,:,:),number_pilot/4,1);repmat(pilot(4,:,:),number_pilot/4,1)];
    end
elseif k==8
    pilot_signal=pilot;
elseif k==16
    pilot_signal=pilot;
elseif k==32
    pilot_signal=pilot;
end
    

ber_result_smi=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%BER����l�L�^������
ber_result_ccm=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%BER����l�L�^������
ber_result_bind=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%BER����l�L�^������
ber_result_zf=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%BER����l�L�^������
thpt_result_smi=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%Thpt����l�L�^������
thpt_result_ccm=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%Thpt����l�L�^������
thpt_result_bind=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%Thpt����l�L�^������
thpt_result_zf=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%Thpt����l�L�^������


for ebno=1:((ebno_max-ebno_min)/ebno_step)+1
    eb_no=(ebno-1)*ebno_step+ebno_min;
    
    modified_eb_no=eb_no+10*(log10(modulation_index)+log10(number_carrier/(number_carrier+guard_interval))+log10(number_symbol/(number_symbol+number_pilot)));%-log10(number_ant));%�A���e�i�Q�C��(�ϒ������ɑ΂���␳)   
%     modified_eb_no=eb_no+10*(log10(modulation_index)+log10(number_carrier/(number_carrier+guard_interval))+log10(number_symbol/(number_symbol+number_pilot))-log10(number_ant));%�A���e�i���K��(�ϒ������ɑ΂���␳)
    
    for re=1:repeat
        %% ���M�@
        disp(['Eb/N0:', num2str(eb_no) ' USER',num2str(k) ' Repeat', num2str(re)]);
        %���M�M���쐬
        [signal,Desired_bit_signal]=signal_generater(modulation_index,number_symbol,number_carrier,number_user,D);
            
        
        %�p�C���b�g�t���������M�M��
        S=[pilot_signal;signal];
        
        %IFFT(freq2time),�K�[�h�C���^�[�o���t��,P/S�ϊ�
        serial_signal_tx=tx_prepare(S,ifft_size,guard_interval,number_user);
        
        
        %% �`���H
        %AWGN
        noise_dis=10.^(-modified_eb_no/20);
        noise=sqrt(1/2)*(randn(number_ant,size(serial_signal_tx,2))+1j*randn(number_ant,size(serial_signal_tx,2))).*noise_dis;
        
%         Pn=10^(-modified_eb_no/10);
        % noise=(randn(number_ant,size(serial_signal_tx,2))+1j*randn(number_ant,size(serial_signal_tx,2)))*sqrt(Pn/2);      %�����Ӗ�(���ȏ�ver)
        
        %���j�A�A���C�A���e�i�t�@�N�^(�X�e�A�����O�x�N�g��)
        if antenna_form==1
            PSI=exp(-1j*2*pi*d*sin(DOAvector*deg2rad));
            array_resp=(ones(number_ant,1)*PSI).^((0:(number_ant-1)).'*ones(1,number_user));
        elseif antenna_form==2
            PSI_x=exp(-1j*2*pi*d*sin(DOAvector_x*deg2rad));
            array_resp_x=(ones(number_ant_x,1)*PSI_x).^((0:(number_ant_x-1)).'*ones(1,number_user));
            PSI_y=exp(-1j*2*pi*d*sin(DOAvector_y*deg2rad));
            array_resp_y=(ones(number_ant_y,1)*PSI_y).^((0:(number_ant_y-1)).'*ones(1,number_user));
            array_resp=zeros(number_ant,number_user);
            na=1;
            for ny=1:number_ant_y
                for nx=1:number_ant_x
                    array_resp(na,:)=array_resp_x(nx,:).*array_resp_y(ny,:);
                    na=na+1;
                end
            end
        end
        
        faded_Rayleigh=zeros(number_ant,size(serial_signal_tx,2));
        if path_index==1
            %�V���O���p�X
            for nu=1:number_user
                faded_signal(nu,:)=singlepath_LoS(serial_signal_tx(nu,:),trans_rate,Doppler);
            end
            %�M���ɃA���e�i�t�@�N�^������AWGN�t���������̂���M�����(�U����sqrt(ant)�Ŋ����Đ��K��->�d�͂�1/ant)
            serial_signal_rx=(array_resp*faded_signal+noise);%./sqrt(number_ant);
            
        elseif path_index==2
            %�}���`�p�X
            for nu=1:number_user
                faded_signal(nu,:)=multipath(serial_signal_tx(nu,:),trans_rate,Doppler,number_path,decay_factor,path_interval);
            end    
            %�M���ɃA���e�i�t�@�N�^������AWGN�t���������̂���M�����(�U����sqrt(ant)�Ŋ����Đ��K��->�d�͂�1/ant)
            serial_signal_rx=(array_resp*faded_signal+noise);%./sqrt(number_ant);
            
        elseif path_index==3%%%%%%���������Ċ֐������Ă���multipath��1�p�X�߂�LoS��path_amplitude�Ɠ��B�^�C�~���O�����ɂȂ�H�H�H�H
            %LoS
            for nu=1:number_user
%                 frame=71.4*10^(-6);     %LTE,5G
                frame=4*10^(-6);     %802.11
                t=frame/(guard_interval+number_carrier); %1�V���{������?
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 �S���Œx��
                time=[0:t:frame*(number_pilot+number_symbol)-t];
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                tx_fading=serial_signal_tx.*exp(1j*2*pi*Doppler*repmat(time,number_user,1));    %�S�������ړ��i���x�͂Ƃ������������j�ɂȂ��Ă���Hmultipath�̂Ƃ��͊֌W�Ȃ��������ǂ����̂��c�H
            end
%             faded_LoS=array_resp*serial_signal_tx;
            faded_LoS=array_resp*tx_fading;
            %multipath(LoS�����ăp�X���ɂȂ�悤��number_path-1�ɂ��Ă���)
            for na=1:number_ant
                for nu=1:number_user
                    faded_Rayleigh(na,:)=faded_Rayleigh(na,:)+multipath(serial_signal_tx(nu,:),trans_rate,Doppler,number_path-1,decay_factor,path_interval);
                end
            end
            faded_signal=sqrt(K/(1+K))*faded_LoS+sqrt(1/(1+K))*faded_Rayleigh;
            serial_signal_rx=(faded_signal+noise);%./sqrt(number_ant);
        end
        
        
        
        
        %% ��M�@
        %S/P�ϊ�,�K�[�h�C���^�[�o������,FFT(time2freq),���בւ�
        fft_sig=rx_prepare(serial_signal_rx,ifft_size,guard_interval,number_ant,number_carrier,number_symbol,number_pilot);
        
        %�`���l���s��v�Z%
        H_resp=fft_sig(:,1:number_pilot,:);
        %pilot���ω�
        Hresp=zeros(number_ant,number_user,number_carrier);
        for j=1:size(Hresp,2)
            Hresp(:,j,:)=sum(H_resp(:,number_pilot*(j-1)/number_user+1:number_pilot*j/number_user,:),2)/(number_pilot/number_user);
        end

        for nc=1:number_carrier
            ref(:,:,nc)=Hresp(:,:,nc)*inv(hadamard_m);
        end
        
        %% ���o�@
        %���M�M��S(�V���{���~�L�����A�~���[�U)->signal_tx(Desired_User,�V���{���~�L�����A)�ɕϊ�
        signal_tx=reshape(S(:,:,D),1,number_symbol+number_pilot,number_carrier);
        %signal_tx�̃p�C���b�g���������o��
        pilot_tx=signal_tx(:,1:number_pilot,:);
        
        for nc=1:number_carrier
            Xt=fft_sig(:,:,nc);
            %���ω�
%             Xtave=Hresp(:,:,nc);

            
           %% SMI%
            %�����𐄒�
            %���֍s��
            %������
            Psmi=Xt(:,1:number_pilot)*Xt(:,1:number_pilot)'/number_pilot;
%             Psmi=Xtave*Xtave'/number_user;
            %���H
%             Psmi=Xt*Xt'/(number_symbol+number_pilot);
            %���փx�N�g��
            Vsmi=Xt(:,1:number_pilot)*conj(pilot_signal(:,nc,D))/number_pilot;
%             Vsmi=Xtave*conj(pilot_signal(1:4,nc,D))/number_user;
            %�d�݌v�Z
            Ws=pinv(Psmi)*Vsmi;  
%             Ws=pinv(Psmi)*ref(:,D,nc);  
            Wsmi=(Ws./(Vsmi'*Ws));
            
            
            %compensation(�⏞)
            Rec_smi(nc,:)=Wsmi'*Xt;
            
            
%             %% ZF(MRC)
%             %compensation
%             H=ref(:,:,nc)*ref(:,:,nc)';
%             Rec_alluser=H*Xt;
% %             Rec_alluser=pinv(ref(:,:,nc))*Xt;
%             Rec_zf(nc,:)=Rec_alluser(D,:);
            
        end
        
        %% bind SMI(���݂ɓ����)
        nc=1;
        while nc<=64
            for number_bind_tmp=[1,2,4,8,16,32,64]
                %         for nc=1:number_bind:number_carrier
                %number_bind�{���˂��Ƃ�
                Xbind_tmp=zeros(number_ant,number_pilot*number_bind_tmp);
                cnt=1;
                for np=1:number_pilot
                    for nb=1:number_bind_tmp
                        carrier=nc+nb-1;
                        if carrier>64
                            carrier=64;
                        end
                        Xbind_tmp(:,cnt)=fft_sig(:,np,carrier);
%                         if nc==64 && carrier==64
%                             break;      %%%%�v�C��(nc=64�̂Ƃ�carrier64��cnt�������Ȃ��̂�Xbind_tmp1��ڂɑS���㏑������Ă��܂��A���̓������)
%                         end
                        cnt=cnt+1;
                    end
                 end
                Xbind_tmp_abs=abs(Xbind_tmp);
                Xbind_tmp_std=zeros(number_ant,number_pilot);
                for np=1:number_pilot
                    Xbind_tmp_std(:,np)=std(Xbind_tmp_abs(:,(np-1)*number_bind_tmp+1:np*number_bind_tmp),0,2);
                end
%                 Xbind_tmp_std_mean=mean(Xbind_tmp_std,'all');%%%���ςł������͋ᖡ�̗]�n����i�덷������ő�l��]����ɂ���Ƃ��j
                Xbind_tmp_std_mean=mean(mean(Xbind_tmp_std),2);
                if Xbind_tmp_std_mean > sigma
                    break;
                elseif nc+number_bind_tmp*2-1 > 64%nc==64 && carrier==64    %�T�u�L�����A������������������(�ŏI�T�u�L�����A�܂ł����Ă��܂�����)bind�I���
                    number_bind_tmp=number_bind_tmp*2;
                    break;
                end
            end
            %%%%%%%%    break������g������Xbind�͏㏑�����ꂿ����Ă邩��������v�Z����A���Ƃ͂����nc���[�v�̒��ɂ��܂����ꍞ��
            
            number_bind=number_bind_tmp/2;      %���ꂾ��64�ɂ͂Ȃ�Ȃ����ǁA���p�セ��ȑ傫�Ȗ��ł͂Ȃ����炢�����B
            Xbind=zeros(number_ant,number_pilot*number_bind);
            cnt=1;
            for np=1:number_pilot
                for nb=1:number_bind
                    carrier=nc+nb-1;
                    if carrier>64
                        carrier=64;
                    end
                    Xbind(:,cnt)=fft_sig(:,np,carrier);
                    cnt=cnt+1;
                end
            end
            
            
            %bind���ω�   %��bind�����ω��������Ă݂����ǂ��܂�悭�Ȃ��H�H(313~317)<-�����s��ɂȂ��Ă��܂��Ƃ������邩��H
            Xbind_ave=zeros(number_ant,number_pilot);
            for i=1:size(Xbind_ave,2)
                Xbind_ave(:,i)=sum(Xbind(:,(i-1)*number_bind+1:i*number_bind),2)/number_bind;
            end
%             %             %pilot���ω�
%             %             Xbindave=zeros(number_ant,number_user);
%             %             for j=1:size(Xbindave,2)
%             %                 Xbindave(:,j)=sum(Xbind_ave(:,number_pilot*(j-1)/number_user+1:number_pilot*j/number_user),2)/(number_pilot/number_user);
%             %             end
%             
%             %             Pbind=Xbindave*Xbindave'/number_user;
            Pbind=Xbind_ave*Xbind_ave'/number_pilot;
%             Pbind=Xbind*Xbind'/(number_pilot*number_bind);
            Vbind=Xbind*conj(repmat(pilot_signal(:,nc,D),number_bind,1))/(number_pilot*number_bind);        %D=1����Ȃ��Ƃ��͏��������K�v�ȋC������(repmat)
            Wb=pinv(Pbind)*Vbind;
            Wbind=(Wb./(Vbind'*Wb));
            
            %compensation(�⏞)
            for nb=1:number_bind
                carrier=nc+nb-1;
                if carrier>64
                    carrier=64;
                end
                Rec_bind(nc+nb-1,:)=Wbind'*fft_sig(:,:,carrier);
            end
            
            nc=nc+number_bind;
        end
        
        
        %% CCM-SMI%
        %���֍s��
        %������
        Pccm=serial_signal_rx(:,1:((guard_interval+number_carrier)*number_pilot))*serial_signal_rx(:,1:((guard_interval+number_carrier)*number_pilot))'/((guard_interval+number_carrier)*number_pilot);
        %���H
%         Pccm=serial_signal_rx*serial_signal_rx'/((guard_interval+number_carrier)*(number_symbol+number_pilot));
        %���փx�N�g��
        Vccm=serial_signal_rx(:,1:((guard_interval+number_carrier)*number_pilot))*conj(serial_signal_tx(D,1:(guard_interval+number_carrier)*number_pilot).')/((guard_interval+number_carrier)*number_pilot);
        %�d��
        Wc=pinv(Pccm)*Vccm;
        Wccm=(Wc./(Vccm'*Wc));
        
        %compensation(�⏞)
        for nc=1:number_carrier
            Rec_ccm(nc,:)=Wccm'*fft_sig(:,:,nc);
        end

        
        %% ����,���v�Z
%         
%         W(:,1)=Wsmi(:,1);
%         W(:,3)=Wccm(:,1);
        
        %Rec(carrier*symbol)->Recover(symbol*carrier)��
        Recover_smi=Rec_smi.';
        Recover_ccm=Rec_ccm.';
        Recover_bind=Rec_bind.';
%         Recover_zf=Rec_zf.';    %%%%
        
        detected_bit_smi=demodulation(modulation_index,Recover_smi, number_symbol, number_pilot, number_carrier);
        detected_bit_ccm=demodulation(modulation_index,Recover_ccm, number_symbol, number_pilot, number_carrier);
        detected_bit_bind=demodulation(modulation_index,Recover_bind, number_symbol, number_pilot, number_carrier);
%          detected_bit_zf=demodulation(modulation_index,Recover_zf, number_symbol, number_pilot, number_carrier);    %%%%
        
        %BER�v�Z
        ber_result_smi=ber_calculation(modulation_index,ber_result_smi, Desired_bit_signal, detected_bit_smi, ebno, modified_eb_no);
        ber_result_ccm=ber_calculation(modulation_index,ber_result_ccm, Desired_bit_signal, detected_bit_ccm, ebno, modified_eb_no);
        ber_result_bind=ber_calculation(modulation_index,ber_result_bind, Desired_bit_signal, detected_bit_bind, ebno, modified_eb_no);
%         ber_result_zf=ber_calculation(modulation_index,ber_result_zf, Desired_bit_signal, detected_bit_zf, ebno, modified_eb_no);   %%%%

        %Thpt�v�Z
        thpt_result_smi=thpt_calculation(modulation_index, thpt_result_smi, Desired_bit_signal, detected_bit_smi, ebno, modified_eb_no, trans_rate, serial_signal_tx);
        thpt_result_ccm=thpt_calculation(modulation_index, thpt_result_ccm, Desired_bit_signal, detected_bit_ccm, ebno, modified_eb_no, trans_rate, serial_signal_tx);
        thpt_result_bind=thpt_calculation(modulation_index, thpt_result_bind, Desired_bit_signal, detected_bit_bind, ebno, modified_eb_no, trans_rate, serial_signal_tx);
%         thpt_result_zf=thpt_calculation(modulation_index, thpt_result_zf, Desired_bit_signal, detected_bit_proccm, ebno, modified_eb_no, trans_rate, serial_signal_tx);
    end
end
%% �v���b�g
ber_result_smi(:,2)=ber_result_smi(:,2)./repeat;
ber_result_ccm(:,2)=ber_result_ccm(:,2)./repeat;
ber_result_bind(:,2)=ber_result_bind(:,2)./repeat;
% ber_result_zf(:,2)=ber_result_zf(:,2)./repeat;  %%%%

figure(1);
semilogy(ebno_min:ebno_step:ebno_max,ber_result_smi(:,2)','ro--');
hold on;
semilogy(ebno_min:ebno_step:ebno_max,ber_result_ccm(:,2)','b<-');
semilogy(ebno_min:ebno_step:ebno_max,ber_result_bind(:,2)','g^-');
% semilogy(ebno_min:ebno_step:ebno_max,ber_result_zf(:,2)','kx-');    %%%%

legend('MMSE-SMI' , 'CCM-SMI','bind');
% legend('MMSE-SMI' , 'CCM-SMI','bind','MRC');

title('BER performance');
xlabel('Eb/No [dB]');
ylabel('BER');
axis([0,15,10^(-5),10^(0)]);
grid on;

%Thpt�v���b�g
thpt_result_smi(:,2)=thpt_result_smi(:,2)./repeat;
thpt_result_ccm(:,2)=thpt_result_ccm(:,2)./repeat;
thpt_result_bind(:,2)=thpt_result_bind(:,2)./repeat;
% thpt_result_zf(:,2)=thpt_result_zf(:,2)./repeat; 

figure(2);
plot(ebno_min:ebno_step:ebno_max,thpt_result_smi(:,2)','ro-');
hold on;
plot(ebno_min:ebno_step:ebno_max,thpt_result_ccm(:,2)','b^-');
plot(ebno_min:ebno_step:ebno_max,thpt_result_bind(:,2)','g<-');

legend('MMSE-SMI' , 'CCM-SMI','bind');
title('Throughput');
xlabel('Eb/No [dB]');
ylabel('Throughput');
axis([0,15,0,10^7]);
grid on;


%BER�ۑ�
% saveas(gcf,num2str(K_dB));
% saveas(gcf,'K50fd001sigma02');


end



% %�A���e�i�p�^�[��%(30dB�̂Ƃ�.repeat�����ω�.SMI��carrier64�̂Ƃ�)
% WW=W./repeat;
% doa=-90:90;     %Degree of Arrival
% figure(2);
% ant_pattern(doa,d,number_ant,WW(:,4));      %proccm�̃A���e�i�p�^�[��
% title('Prop.CCM-SMI');
% figure(3);
% ant_pattern(doa,d,number_ant,WW(:,3));      %ccm�̃A���e�i�p�^�[��
% title('Conv.CCM-SMI');
% figure(4);
% ant_pattern(doa,d,number_ant,WW(:,2));      %prosmi�̃A���e�i�p�^�[��
% title('Prop.SMI');
% figure(5);
% ant_pattern(doa,d,number_ant,WW(:,1));      %smi�̃A���e�i�p�^�[��
% title('Conv.SMI');
% 
% figure(6);
% plot(Recover_ccm(number_pilot+1:number_pilot+number_symbol,:),'.');
% title('Conv.CCM-SMI');
% axis([-1.5,1.5,-1.5,1.5]);
% grid on;

% figure(9);
% plot(Recover_smi(number_pilot+1:number_pilot+number_symbol,:),'.');
% title('MMSE-SMI');
% axis([-1.5,1.5,-1.5,1.5]);
% grid on;
% 
% 
% figure(10);
% plot(Recover_bind(number_pilot+1:number_pilot+number_symbol,:),'.');
% title('bind');
% axis([-1.5,1.5,-1.5,1.5]);
% grid on;


tocf(3);